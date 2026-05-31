//! Parsing and navigation of VCS merge conflict markers.
//!
//! Supports three conflict formats:
//!
//! **git 2-way:**
//! ```text
//! <<<<<<< current
//! ... current text ...
//! =======
//! ... incoming text ...
//! >>>>>>> incoming
//! ```
//!
//! **git diff3 (3-way):**
//! ```text
//! <<<<<<< current
//! ... current text ...
//! ||||||| base
//! ... base text ...
//! =======
//! ... incoming text ...
//! >>>>>>> incoming
//! ```
//!
//! **jj snapshot (N-way):**
//! ```text
//! <<<<<<< conflict
//! +++++++ side #1
//! ... side 1 text ...
//! ------- base
//! ... base text ...
//! +++++++ side #2
//! ... side 2 text ...
//! >>>>>>> conflict ends
//! ```
//!
//! All three formats are unified into [`ConflictRegion`] with a `Vec<Section>`.

use std::collections::HashMap;
use std::ops::Range;

use imara_diff::{Algorithm, Diff, InternedInput};

use crate::Rope;

// ── Types ─────────────────────────────────────────────────────────────────────

/// Pair of removed/added word-diff ranges for a conflict refine pair.
type RefineDiffs = (Vec<Range<usize>>, Vec<Range<usize>>);

/// Per-conflict word-level diff cache entry, keyed by conflict start position.
#[derive(Debug, Clone)]
pub struct ConflictRefineEntry {
    /// Current refine pair index (see [`ConflictRegion::refine_pair_indices`]).
    /// Set to `num_refine_pairs()` to show no word-level highlights.
    pub pair: usize,
    /// When `true`, show all adjacent (Side, Base) pair word-diffs
    /// simultaneously as the default state (before any single-pair cycle).
    pub show_base_pairs: bool,
    /// Cached word-diff results for the current `pair`:
    /// `(removed_ranges, added_ranges)`.
    pub diffs: Option<RefineDiffs>,
    /// Cached added-word ranges for Side sections vs resolved Diff base.
    /// Indexed by section position in the region. `None` = not computed yet.
    pub side_added: Option<Vec<Vec<Range<usize>>>>,
}

impl Default for ConflictRefineEntry {
    fn default() -> Self {
        Self {
            pair: 0,
            show_base_pairs: true,
            diffs: None,
            side_added: None,
        }
    }
}

/// Per-conflict word-diff refine state, keyed by [`ConflictRegion::start`].
///
/// Cleared on every edit; the active pair setting (`entry.pair`) is preserved
/// when the cursor was inside a conflict before the edit.
pub type ConflictCache = HashMap<usize, ConflictRefineEntry>;

/// Whether a section holds one side of a conflict, a common base, or a
/// jj unified-diff side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    /// A `+++++++` side (jj format) or the current/incoming sides (git format).
    Side,
    /// A `-------` base (jj format) or the `|||||||` base (git diff3 format).
    Base,
    /// A `%%%%%%%` diff side (jj format) — a unified diff against the merge base.
    Diff,
}

/// One section within a conflict region.
///
/// For git format the first section's `marker_start` equals the `<<<<<<<` line;
/// for jj format every section starts with its own `+++++++` / `-------` / `%%%%%%%` marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub kind: SectionKind,
    /// Char index of the first character of this section's first marker line.
    ///
    /// For the first side of a git-format conflict this is the `<<<<<<<` line.
    /// For jj-format sections this is the `+++++++`, `-------`, or `%%%%%%%` line.
    /// For the last side of a git-format conflict this is the `=======` line.
    pub marker_start: usize,
    /// How many consecutive marker lines this section has (1 for most, 2 for
    /// a `%%%%%%%` Diff section whose second line is `\\\\\\`).
    pub marker_lines: usize,
    /// Char index of the first content character (the line *after* the last
    /// marker line).  For `Diff` sections this is after the `\\\\\\` continuation.
    pub content_start: usize,
    /// Exclusive end of the content (= `marker_start` of next section, or `end`
    /// of the whole conflict for the final section).
    pub content_end: usize,
}

/// A single VCS merge conflict region found in a document.
///
/// All positions are **character indices** into the document rope.
/// `start` points to the first character of the `<<<<<<<` marker line.
/// `end` points to the character just past the last character of the `>>>>>>>`
/// line (i.e. after the trailing newline, if present).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictRegion {
    /// Start of the `<<<<<<< ...` line.
    pub start: usize,
    /// Ordered list of conflict sections.  There is always at least one `Side`.
    pub sections: Vec<Section>,
    /// Exclusive end past the `>>>>>>> ...` line.
    pub end: usize,
}

impl ConflictRegion {
    /// Total number of single‑pair states available via refine.
    ///
    /// These are the states reachable *after* the initial `show_base_pairs`
    /// step.  The count is:
    ///
    ///   base_side (0 or 1)  +  Side‑Side upper‑triangular over non‑Base
    ///
    /// where `base_side` covers the last‑Base‑vs‑last‑Side pair (Phase 1 of
    /// [`refine_pair_indices`]) and the upper‑triangular covers all remaining
    /// Non‑Base × Non‑Base combinations (Phase 2).
    pub fn num_refine_pairs(&self) -> usize {
        let non_base_count = self
            .sections
            .iter()
            .filter(|s| s.kind != SectionKind::Base)
            .count();
        let side_side = if non_base_count < 2 {
            0
        } else {
            non_base_count * (non_base_count - 1) / 2
        };
        let has_base = self.sections.iter().any(|s| s.kind == SectionKind::Base);
        let last_is_side = self
            .sections
            .last()
            .is_some_and(|s| s.kind == SectionKind::Side);
        let base_side = if has_base && last_is_side { 1 } else { 0 };
        base_side + side_side
    }

    /// Return the pair of section indices for the given refine `pair` index.
    ///
    /// Returns `None` when `pair >= num_refine_pairs()`.
    ///
    /// Ordering (two phases):
    ///
    /// 1. **Base–Side** — pair `0` (when a Base exists and the last section
    ///    is a Side): the last Base section vs the last Side section.
    /// 2. **Non‑Base × Non‑Base** — pairs `1..N` in upper‑triangular order
    ///    over all sections that are *not* Base (Side and Diff).
    pub fn refine_pair_indices(&self, pair: usize) -> Option<(usize, usize)> {
        let non_base: Vec<usize> = (0..self.sections.len())
            .filter(|i| self.sections[*i].kind != SectionKind::Base)
            .collect();
        let m = non_base.len();
        let side_side = m * m.saturating_sub(1) / 2;
        let has_base = self.sections.iter().any(|s| s.kind == SectionKind::Base);
        let last_is_side = self
            .sections
            .last()
            .is_some_and(|s| s.kind == SectionKind::Side);
        let base_side = if has_base && last_is_side { 1 } else { 0 };
        let total = base_side + side_side;
        if pair >= total {
            return None;
        }
        // Phase 1: last Base vs last Side (when available, at index 0)
        if base_side > 0 && pair == 0 {
            let last_side_idx = self.sections.len() - 1;
            let last_base_idx = self.sections[..last_side_idx]
                .iter()
                .rposition(|s| s.kind == SectionKind::Base)?;
            return Some((last_base_idx, last_side_idx));
        }
        // Remaining pairs are shifted by `base_side`.
        // Phase 2: upper‑triangular over non‑Base sections
        let offset = base_side;
        let ss_pair = pair - offset;
        let mut idx = 0;
        for ai in 0..m {
            for bi in (ai + 1)..m {
                if idx == ss_pair {
                    return Some((non_base[ai], non_base[bi]));
                }
                idx += 1;
            }
        }
        None
    }

    /// Whether this conflict has at least one adjacent (Side, Base) pair
    /// — two consecutive sections where the first is Side and the second is
    /// Base.  These are the pairs shown in the initial `show_base_pairs`
    /// state.
    pub fn has_adjacent_side_base_pairs(&self) -> bool {
        self.sections
            .windows(2)
            .any(|w| w[0].kind == SectionKind::Side && w[1].kind == SectionKind::Base)
    }

    /// Whether this conflict has a "base comparison" — a default view shown
    /// before any single‑pair cycle.  True when the conflict has adjacent
    /// (Side, Base) pairs (git diff3 / jj snapshot) or Diff sections (jj
    /// diff format, where the always‑on highlights form the base comparison).
    pub fn has_base_comparison(&self) -> bool {
        self.has_adjacent_side_base_pairs()
            || self.sections.iter().any(|s| s.kind == SectionKind::Diff)
    }

    /// Return the section-index pairs for all adjacent (Side, Base) pairs in
    /// document order.
    pub fn adjacent_side_base_pairs(&self) -> Vec<(usize, usize)> {
        self.sections
            .windows(2)
            .enumerate()
            .filter(|(_, w)| w[0].kind == SectionKind::Side && w[1].kind == SectionKind::Base)
            .map(|(i, _)| (i, i + 1))
            .collect()
    }
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Parse all conflict regions in `text`, in document order.
///
/// Supports git 2-way, git diff3 3-way, and jj snapshot N-way formats.
/// Also recognises "bare" conflicts — a `<<<<<<<` / `>>>>>>>` pair with no
/// inner markers — which arise after manually editing out unwanted sections.
/// Incomplete conflict blocks (no closing `>>>>>>>`) are silently ignored.
pub fn find_conflicts(text: &Rope) -> Vec<ConflictRegion> {
    /// A section whose `content_end` is not yet known.
    #[derive(Debug, Clone)]
    struct PartialSection {
        kind: SectionKind,
        marker_start: usize,
        marker_lines: usize,
        content_start: usize,
    }

    impl PartialSection {
        fn finish(self, content_end: usize) -> Section {
            Section {
                kind: self.kind,
                marker_start: self.marker_start,
                marker_lines: self.marker_lines,
                content_start: self.content_start,
                content_end,
            }
        }
    }

    #[derive(Debug, Default)]
    enum State {
        #[default]
        Idle,
        /// Inside a conflict region (git or jj format).
        InConflict {
            start: usize,
            sections: Vec<Section>,
            current: PartialSection,
            format: ConflictFormat,
        },
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum ConflictFormat {
        Git,
        Jj,
    }

    let mut conflicts: Vec<ConflictRegion> = Vec::new();
    let mut state = State::Idle;
    let mut line_char_start: usize = 0;

    for line in text.lines() {
        let line_len = line.len_chars();
        let next_line_start = line_char_start + line_len;

        let is_marker = |c: char| -> bool {
            let mut chars = line.chars();
            (&mut chars).take(7).all(|ch| ch == c)
                && matches!(chars.next(), None | Some(' ') | Some('\r') | Some('\n'))
        };

        state = match std::mem::take(&mut state) {
            State::Idle => {
                if is_marker('<') {
                    State::InConflict {
                        start: line_char_start,
                        sections: Vec::new(),
                        current: PartialSection {
                            kind: SectionKind::Side,
                            marker_start: line_char_start,
                            marker_lines: 1,
                            content_start: next_line_start,
                        },
                        format: ConflictFormat::Git,
                    }
                } else {
                    State::Idle
                }
            }

            State::InConflict {
                start,
                mut sections,
                mut current,
                format,
            } => {
                if is_marker('|') {
                    sections.push(current.finish(line_char_start));
                    State::InConflict {
                        start,
                        sections,
                        current: PartialSection {
                            kind: SectionKind::Base,
                            marker_start: line_char_start,
                            marker_lines: 1,
                            content_start: next_line_start,
                        },
                        format,
                    }
                } else if is_marker('=') {
                    sections.push(current.finish(line_char_start));
                    State::InConflict {
                        start,
                        sections,
                        current: PartialSection {
                            kind: SectionKind::Side,
                            marker_start: line_char_start,
                            marker_lines: 1,
                            content_start: next_line_start,
                        },
                        format,
                    }
                } else if is_marker('>') {
                    let end = next_line_start;
                    sections.push(current.finish(line_char_start));
                    if !sections.is_empty() {
                        conflicts.push(ConflictRegion {
                            start,
                            sections,
                            end,
                        });
                    }
                    State::Idle
                } else if is_marker('<') {
                    State::InConflict {
                        start: line_char_start,
                        sections: Vec::new(),
                        current: PartialSection {
                            kind: SectionKind::Side,
                            marker_start: line_char_start,
                            marker_lines: 1,
                            content_start: next_line_start,
                        },
                        format: ConflictFormat::Git,
                    }
                } else if is_marker('+') || is_marker('-') {
                    if format == ConflictFormat::Git {
                        if current.content_start < line_char_start {
                            sections.push(current.finish(line_char_start));
                        }
                    } else {
                        sections.push(current.finish(line_char_start));
                    }
                    let kind = if is_marker('+') {
                        SectionKind::Side
                    } else {
                        SectionKind::Base
                    };
                    State::InConflict {
                        start,
                        sections,
                        current: PartialSection {
                            kind,
                            marker_start: line_char_start,
                            marker_lines: 1,
                            content_start: next_line_start,
                        },
                        format: ConflictFormat::Jj,
                    }
                } else if is_marker('%') {
                    if current.content_start < line_char_start {
                        sections.push(current.finish(line_char_start));
                    }
                    State::InConflict {
                        start,
                        sections,
                        current: PartialSection {
                            kind: SectionKind::Diff,
                            marker_start: line_char_start,
                            marker_lines: 1,
                            content_start: next_line_start,
                        },
                        format: ConflictFormat::Jj,
                    }
                } else if is_marker('\\') {
                    if current.kind == SectionKind::Diff {
                        current.content_start = next_line_start;
                        current.marker_lines = 2;
                    }
                    State::InConflict {
                        start,
                        sections,
                        current,
                        format,
                    }
                } else {
                    State::InConflict {
                        start,
                        sections,
                        current,
                        format,
                    }
                }
            }
        };

        line_char_start = next_line_start;
    }

    conflicts
}

// ── Cursor position helpers ───────────────────────────────────────────────────

/// Return the index of the conflict in `conflicts` that contains `char_pos`,
/// or `None` if the position is not inside any conflict region.
pub fn conflict_at(conflicts: &[ConflictRegion], char_pos: usize) -> Option<usize> {
    conflicts
        .iter()
        .position(|c| char_pos >= c.start && char_pos < c.end)
}

/// Return the index of the next conflict after `char_pos` (exclusive).
pub fn next_conflict(conflicts: &[ConflictRegion], char_pos: usize) -> Option<usize> {
    conflicts.iter().position(|c| c.start > char_pos)
}

/// Return the index of the previous conflict before `char_pos` (exclusive).
pub fn prev_conflict(conflicts: &[ConflictRegion], char_pos: usize) -> Option<usize> {
    conflicts.iter().rposition(|c| c.start < char_pos)
}

/// Return which section index of `region` the character position `char_pos` is in,
/// or `None` if it is on a pure separator line (the `=======` line in git format)
/// or outside `region`.
///
/// Marker lines are attributed to their own section (the `<<<<<<<` line is part
/// of the first section, `+++++++`/`-------`/`|||||||` lines are part of their
/// section, `>>>>>>>` is part of the last section).  Only the `=======`
/// separator is a no-op since it doesn't logically belong to either side.
pub fn conflict_section_at(region: &ConflictRegion, text: &Rope, char_pos: usize) -> Option<usize> {
    // For git-format conflicts, the last section's marker is `=======`.
    // A cursor sitting on that marker line belongs to no section (no-op).
    // For jj-format, there is no `=======` — every marker line belongs to its section.
    let sep_line_end = git_sep_line_end(region, text);

    for (i, section) in region.sections.iter().enumerate() {
        let section_end = if i == region.sections.len() - 1 {
            region.end
        } else {
            region.sections[i + 1].marker_start
        };

        if char_pos >= section.marker_start && char_pos < section_end {
            // If this is the last section in git format, the marker line is `=======`
            // and should be a no-op zone.
            if i == region.sections.len() - 1 {
                if let Some(sep_end) = sep_line_end {
                    let sep_start = section.marker_start;
                    if char_pos >= sep_start && char_pos < sep_end {
                        return None; // on the ======= line
                    }
                }
            }
            return Some(i);
        }
    }

    // Shouldn't reach here if char_pos is inside the region.
    None
}

/// For a git-format conflict, return the exclusive end of the `=======` line.
/// Returns `None` for jj-format conflicts (which have no `=======`).
fn git_sep_line_end(region: &ConflictRegion, text: &Rope) -> Option<usize> {
    // In git format, the last section's marker is always `=======`.
    // In jj format, the last section's marker is `+++++++` or `-------`.
    let last = region.sections.last()?;
    // A `=======` line starts with exactly 7 `=` and nothing else (or a space/newline)
    let marker_char: char = text.char(last.marker_start);
    if marker_char == '=' {
        let sep_line = text.char_to_line(last.marker_start);
        Some(text.line_to_char(sep_line + 1))
    } else {
        None
    }
}

/// Returns the document line numbers of every conflict marker line in `text`:
/// the opening `<<<<<<<`, every section marker (`|||||||`, `=======`, `+++++++`,
/// `-------`), and the closing `>>>>>>>`.
///
/// The returned `Vec` is sorted and deduplicated.
pub fn conflict_marker_lines(conflicts: &[ConflictRegion], text: &Rope) -> Vec<usize> {
    let mut lines: Vec<usize> = conflicts
        .iter()
        .flat_map(|region| {
            // Section marker lines (includes the opening <<<<<<< via sections[0].marker_start).
            let section_lines = region.sections.iter().flat_map(|s| {
                let first = text.char_to_line(s.marker_start);
                (0..s.marker_lines).map(move |i| first + i)
            });
            // Closing >>>>>>> line — `end` is exclusive and points past the trailing
            // newline, so subtract 1 to land somewhere on the last line.
            let end_line = std::iter::once(text.char_to_line(region.end.saturating_sub(1)));
            section_lines.chain(end_line)
        })
        .collect();
    lines.sort_unstable();
    lines.dedup();
    lines
}

// ── Content helpers ───────────────────────────────────────────────────────────

/// Return the char range of the first `Side` section ("current" change).
pub fn current_content(region: &ConflictRegion) -> (usize, usize) {
    let s = region
        .sections
        .iter()
        .find(|s| s.kind == SectionKind::Side)
        .expect("conflict must have at least one Side section");
    (s.content_start, s.content_end)
}

/// Return the char range of the last `Side` section ("incoming" change).
pub fn incoming_content(region: &ConflictRegion) -> (usize, usize) {
    let s = region
        .sections
        .iter()
        .rev()
        .find(|s| s.kind == SectionKind::Side)
        .expect("conflict must have at least one Side section");
    (s.content_start, s.content_end)
}

/// Return the char range of the first `Base` section, or `None` if absent.
pub fn base_content(region: &ConflictRegion) -> Option<(usize, usize)> {
    let s = region
        .sections
        .iter()
        .find(|s| s.kind == SectionKind::Base)?;
    Some((s.content_start, s.content_end))
}

/// Return the content of all `Side` sections concatenated (for accept-all).
pub fn all_sides_content(text: &Rope, region: &ConflictRegion) -> String {
    region
        .sections
        .iter()
        .filter(|s| s.kind != SectionKind::Base)
        .map(|s| match s.kind {
            SectionKind::Diff => resolve_diff_content(text, s),
            _ => text.slice(s.content_start..s.content_end).to_string(),
        })
        .collect()
}

/// Strip unified-diff markers from a Diff section, producing the
/// side content (`+` and `   ` lines kept, `-` lines dropped).
pub fn resolve_diff_content(text: &Rope, section: &Section) -> String {
    let mut result = String::new();
    for line in text
        .slice(section.content_start..section.content_end)
        .lines()
    {
        let line_str = line.to_string();
        let first = line_str.chars().next();
        match first {
            Some('-') => {}
            Some('+') | Some(' ') => result.push_str(&line_str[1..]),
            _ => result.push_str(&line_str),
        }
    }
    result
}

/// Strip unified-diff markers from a Diff section, producing the
/// base content (`-` and `   ` lines kept, `+` lines dropped).
pub fn resolve_diff_content_base(text: &Rope, section: &Section) -> String {
    let mut result = String::new();
    for line in text
        .slice(section.content_start..section.content_end)
        .lines()
    {
        let line_str = line.to_string();
        let first = line_str.chars().next();
        match first {
            Some('+') => {}
            Some('-') | Some(' ') => result.push_str(&line_str[1..]),
            _ => result.push_str(&line_str),
        }
    }
    result
}

// ── Refine (word-level diff) ──────────────────────────────────────────────────

/// Returns the content ranges `(left, right)` for refine pair `pair`.
///
/// Pairs are enumerated in upper-triangular order over all non-Base sections.
/// For (Diff, Side) pairs the order is swapped so Side is on the left.
/// Returns `None` if `pair >= region.num_refine_pairs()`.
pub fn conflict_pair_sections(
    region: &ConflictRegion,
    pair: usize,
) -> Option<((usize, usize), (usize, usize))> {
    let (i, j) = region.refine_pair_indices(pair)?;
    // When one section is Diff and the other is Side, put Side on the left.
    let (i, j) = match (region.sections[i].kind, region.sections[j].kind) {
        (SectionKind::Diff, SectionKind::Side) => (j, i),
        _ => (i, j),
    };
    let left = (
        region.sections[i].content_start,
        region.sections[i].content_end,
    );
    let right = (
        region.sections[j].content_start,
        region.sections[j].content_end,
    );
    Some((left, right))
}

/// Look up the active refine pair for a conflict.
///
/// Defaults to `0` if no entry is present.
pub fn conflict_refine_pair(state: &HashMap<usize, usize>, region: &ConflictRegion) -> usize {
    let pair = state.get(&region.start).copied().unwrap_or(0);
    let max = region.num_refine_pairs().saturating_sub(1);
    pair.min(max)
}

// ── Multi-level refinement (jj-style) ─────────────────────────────────────────

/// Whether `c` is a word character.
///
/// Unlike jj's `is_word_byte`, underscore is NOT included so that
/// `snake_case` identifiers split into separate word tokens, giving
/// finer-grained highlighting than jj's default behaviour.
fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c > '\x7F'
}

/// Tokenize a char range by lines. Each line (including trailing newline) becomes
/// one token.
fn tokenize_line_ranges(text: &Rope, range: Range<usize>) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut line_start = range.start;
    for (i, ch) in text.slice(range.clone()).chars().enumerate() {
        let pos = range.start + i;
        if ch == '\n' {
            ranges.push(line_start..pos + 1);
            line_start = pos + 1;
        }
    }
    if line_start < range.end {
        ranges.push(line_start..range.end);
    }
    ranges
}

/// Tokenize a char range by jj-style words.  Only runs of word characters
/// become tokens; non-word characters are skipped.
fn tokenize_word_ranges(text: &Rope, range: Range<usize>) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut word_start: Option<usize> = None;
    for (i, ch) in text.slice(range.clone()).chars().enumerate() {
        let pos = range.start + i;
        if is_word_char(ch) {
            if word_start.is_none() {
                word_start = Some(pos);
            }
        } else if let Some(start) = word_start.take() {
            ranges.push(start..pos);
        }
    }
    if let Some(start) = word_start {
        ranges.push(start..range.end);
    }
    ranges
}

/// Tokenize a char range into individual non-word characters.
/// Word characters are skipped — they were handled at the word level.
fn tokenize_nonword_ranges(text: &Rope, range: Range<usize>) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    for (i, ch) in text.slice(range.clone()).chars().enumerate() {
        let pos = range.start + i;
        if !is_word_char(ch) {
            ranges.push(pos..pos + 1);
        }
    }
    ranges
}

/// A generic token source backed by a `Rope` and a pre-computed range list.
struct RopeTokenSource<'a> {
    text: &'a Rope,
    ranges: Vec<Range<usize>>,
}

impl<'a> imara_diff::TokenSource for RopeTokenSource<'a> {
    type Token = String;
    type Tokenizer = Box<dyn Iterator<Item = String> + 'a>;

    fn tokenize(&self) -> Self::Tokenizer {
        let text = self.text;
        let ranges = self.ranges.clone();
        Box::new(ranges.into_iter().map(move |r| text.slice(r).to_string()))
    }

    fn estimate_tokens(&self) -> u32 {
        self.ranges.len() as u32
    }
}

/// Run a single-level histogram diff of `left` vs `right` within `text`.
///
/// Each matching token pair yields its own (left, right) char-range pair.
/// This is critical when tokens are non-contiguous (e.g. nonword level
/// skips word chars), so that each matched token stands alone and does
/// not also cover the skipped characters between tokens.
fn single_level_diff(
    text: &Rope,
    left: Range<usize>,
    right: Range<usize>,
    tokenizer: impl Fn(&Rope, Range<usize>) -> Vec<Range<usize>>,
) -> Vec<(Range<usize>, Range<usize>)> {
    let left_ranges = tokenizer(text, left);
    let right_ranges = tokenizer(text, right);

    if left_ranges.is_empty() || right_ranges.is_empty() {
        return vec![];
    }

    let left_source = RopeTokenSource {
        text,
        ranges: left_ranges.clone(),
    };
    let right_source = RopeTokenSource {
        text,
        ranges: right_ranges.clone(),
    };

    let input = InternedInput::new(left_source, right_source);
    let diff = Diff::compute(Algorithm::Histogram, &input);

    let mut matches: Vec<(Range<usize>, Range<usize>)> = Vec::new();
    let mut left_idx = 0usize;
    let mut right_idx = 0usize;

    for hunk in diff.hunks() {
        let l_start = hunk.before.start as usize;
        let l_end = hunk.before.end as usize;
        let r_start = hunk.after.start as usize;
        let r_end = hunk.after.end as usize;

        // Emit one pair per matching token before this hunk.
        while left_idx < l_start && right_idx < r_start {
            matches.push((
                left_ranges[left_idx].clone(),
                right_ranges[right_idx].clone(),
            ));
            left_idx += 1;
            right_idx += 1;
        }

        left_idx = l_end;
        right_idx = r_end;
    }

    // Remaining matching tokens after the last hunk.
    while left_idx < left_ranges.len() && right_idx < right_ranges.len() {
        matches.push((
            left_ranges[left_idx].clone(),
            right_ranges[right_idx].clone(),
        ));
        left_idx += 1;
        right_idx += 1;
    }

    matches
}

/// Refine existing matches by re-diffing each gap at a finer granularity.
/// Returns a new match list with any sub-matches inserted.
fn refine_matches(
    text: &Rope,
    matches: &[(Range<usize>, Range<usize>)],
    tokenizer: impl Fn(&Rope, Range<usize>) -> Vec<Range<usize>>,
) -> Vec<(Range<usize>, Range<usize>)> {
    let mut refined = Vec::with_capacity(matches.len());
    refined.push(matches[0].clone());

    for window in matches.windows(2) {
        let (prev_left, prev_right) = &window[0];
        let (next_left, next_right) = &window[1];

        if prev_left.end < next_left.start && prev_right.end < next_right.start {
            let gap_left = prev_left.end..next_left.start;
            let gap_right = prev_right.end..next_right.start;
            refined.extend(single_level_diff(text, gap_left, gap_right, &tokenizer));
        }

        refined.push((next_left.clone(), next_right.clone()));
    }

    refined
}

/// A word-token source for lines in a Diff section matching a specific prefix.
///
/// Only lines whose first character equals `prefix` (typically `'-'` or `'+'`)
/// are tokenized; the prefix character itself is skipped.  Positions are
/// recorded as byte offsets into the original document.
struct DiffSectionWords<'a> {
    text: &'a Rope,
    positions: Vec<(usize, usize)>,
}

impl<'a> DiffSectionWords<'a> {
    fn new(text: &'a Rope, section: &Section, prefix: char) -> Self {
        let mut positions = Vec::new();
        let content = text.slice(section.content_start..section.content_end);
        let mut offset = section.content_start;
        for line in content.lines() {
            let line_len = line.len_chars();
            let first = line.chars().next();
            if first == Some(prefix) {
                let mut tok_start: Option<usize> = None;
                for (i, ch) in line.chars().enumerate() {
                    let pos = offset + i;
                    if i == 0 {
                        continue;
                    }
                    if ch.is_whitespace() {
                        if let Some(s) = tok_start.take() {
                            positions.push((s, pos));
                        }
                    } else if tok_start.is_none() {
                        tok_start = Some(pos);
                    }
                }
                if let Some(s) = tok_start {
                    positions.push((s, offset + line_len));
                }
            }
            offset += line_len;
        }
        Self { text, positions }
    }
}

impl<'a> imara_diff::TokenSource for DiffSectionWords<'a> {
    type Token = String;
    type Tokenizer = Box<dyn Iterator<Item = String> + 'a>;

    fn tokenize(&self) -> Self::Tokenizer {
        let text = self.text;
        let positions = self.positions.clone();
        Box::new(
            positions
                .into_iter()
                .map(move |(s, e)| text.slice(s..e).chars().collect()),
        )
    }

    fn estimate_tokens(&self) -> u32 {
        self.positions.len() as u32
    }
}

/// Run word-level diff between `-` lines and `+` lines within a Diff section.
///
/// Returns `(removed_ranges, added_ranges)` with byte positions in the
/// original document.  Context lines (` ` prefix) are ignored.
pub fn refine_diff_section(
    text: &Rope,
    section: &Section,
) -> (Vec<Range<usize>>, Vec<Range<usize>>) {
    let removed_tokens = DiffSectionWords::new(text, section, '-');
    let added_tokens = DiffSectionWords::new(text, section, '+');

    let removed_positions = removed_tokens.positions.clone();
    let added_positions = added_tokens.positions.clone();

    let input = InternedInput::new(removed_tokens, added_tokens);
    let diff = Diff::compute(Algorithm::Histogram, &input);

    let mut removed: Vec<Range<usize>> = Vec::new();
    let mut added: Vec<Range<usize>> = Vec::new();

    for hunk in diff.hunks() {
        for &(s, e) in &removed_positions[hunk.before.start as usize..hunk.before.end as usize] {
            match removed.last_mut() {
                Some(r) if r.end == s => r.end = e,
                _ => removed.push(s..e),
            }
        }
        for &(s, e) in &added_positions[hunk.after.start as usize..hunk.after.end as usize] {
            match added.last_mut() {
                Some(r) if r.end == s => r.end = e,
                _ => added.push(s..e),
            }
        }
    }

    (removed, added)
}

/// Compute a jj-style multi-level diff between two sections of `text`.
///
/// First diffs by lines, then refines changed regions at word granularity,
/// then at non-word character granularity.  Returns `(removed_ranges,
/// added_ranges)` as half-open char-index intervals.
pub fn refine_diff(
    text: &Rope,
    left: (usize, usize),
    right: (usize, usize),
) -> (Vec<Range<usize>>, Vec<Range<usize>>) {
    // Level 1: line-level diff
    let raw = single_level_diff(text, left.0..left.1, right.0..right.1, tokenize_line_ranges);

    // Build match list with empty boundary sentinels.
    let mut matches: Vec<(Range<usize>, Range<usize>)> = Vec::with_capacity(raw.len() + 2);
    matches.push((left.0..left.0, right.0..right.0));
    matches.extend(raw);
    matches.push((left.1..left.1, right.1..right.1));

    // Level 2: word-level refinement
    matches = refine_matches(text, &matches, tokenize_word_ranges);

    // Level 3: nonword-level refinement
    matches = refine_matches(text, &matches, tokenize_nonword_ranges);

    // Convert gaps between matches to removed/added ranges.
    let mut removed: Vec<Range<usize>> = Vec::new();
    let mut added: Vec<Range<usize>> = Vec::new();
    for window in matches.windows(2) {
        let (prev_left, prev_right) = &window[0];
        let (next_left, next_right) = &window[1];

        if prev_left.end < next_left.start {
            removed.push(prev_left.end..next_left.start);
        }
        if prev_right.end < next_right.start {
            added.push(prev_right.end..next_right.start);
        }
    }

    (removed, added)
}

// ── String-backed word tokens for base-vs-side word-diff ───────────────────────

/// A word-token source backed by a `&str`.
///
/// Tokens are non-whitespace runs of characters.  Positions are byte offsets
/// in the string.  Used as the left (base) side of a word-diff against a
/// Side section in the rope.
struct StringWordTokens<'a> {
    text: &'a str,
    positions: Vec<(usize, usize)>,
}

impl<'a> StringWordTokens<'a> {
    fn new(text: &'a str) -> Self {
        let mut positions = Vec::new();
        let mut tok_start: Option<usize> = None;
        for (i, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if let Some(s) = tok_start.take() {
                    positions.push((s, i));
                }
            } else if tok_start.is_none() {
                tok_start = Some(i);
            }
        }
        if let Some(s) = tok_start {
            positions.push((s, text.len()));
        }
        Self { text, positions }
    }
}

impl<'a> imara_diff::TokenSource for StringWordTokens<'a> {
    type Token = String;
    type Tokenizer = Box<dyn Iterator<Item = String> + 'a>;

    fn tokenize(&self) -> Self::Tokenizer {
        let text = self.text;
        let positions = self.positions.clone();
        Box::new(
            positions
                .into_iter()
                .map(move |(s, e)| text[s..e].to_string()),
        )
    }

    fn estimate_tokens(&self) -> u32 {
        self.positions.len() as u32
    }
}

/// Word-diff a Side section against a resolved base string.
///
/// Returns char-index ranges in the original rope for words that appear in
/// the Side section but not in the base content (added words).  Removed
/// words (present in base, not in Side) are not returned.
pub fn refine_side_with_base(
    text: &Rope,
    side_section: &Section,
    base_content: &str,
) -> Vec<Range<usize>> {
    let side_str = text
        .slice(side_section.content_start..side_section.content_end)
        .to_string();

    let left_tokens = StringWordTokens::new(base_content);
    let right_tokens = StringWordTokens::new(&side_str);

    let right_positions = right_tokens.positions.clone();

    let input = InternedInput::new(left_tokens, right_tokens);
    let diff = Diff::compute(Algorithm::Histogram, &input);

    let mut added_byte: Vec<Range<usize>> = Vec::new();
    for hunk in diff.hunks() {
        for &(s, e) in &right_positions[hunk.after.start as usize..hunk.after.end as usize] {
            match added_byte.last_mut() {
                Some(r) if r.end == s => r.end = e,
                _ => added_byte.push(s..e),
            }
        }
    }

    // Map byte offsets in side_str to char offsets in the rope.
    added_byte
        .into_iter()
        .map(|range| {
            let start = side_section.content_start + side_str[..range.start].chars().count();
            let end = side_section.content_start + side_str[..range.end].chars().count();
            start..end
        })
        .collect()
}

// ── Side–Diff word-diff ────────────────────────────────────────────────────────

/// Like `resolve_diff_content` but also returns a position map: for each char
/// index in the resolved string, the corresponding char index in the rope.
fn resolve_diff_content_mapped(text: &Rope, section: &Section) -> (String, Vec<usize>) {
    let mut content = String::new();
    let mut map = Vec::new();
    let mut rope_char_pos = section.content_start;
    for line in text
        .slice(section.content_start..section.content_end)
        .lines()
    {
        let line_str = line.to_string();
        let first = line_str.chars().next();
        match first {
            Some('-') => {
                rope_char_pos += line_str.chars().count();
            }
            Some('+') | Some(' ') => {
                rope_char_pos += 1; // skip the prefix char
                for c in line_str[1..].chars() {
                    content.push(c);
                    map.push(rope_char_pos);
                    rope_char_pos += 1;
                }
            }
            _ => {
                for c in line_str.chars() {
                    content.push(c);
                    map.push(rope_char_pos);
                    rope_char_pos += 1;
                }
            }
        }
    }
    (content, map)
}

/// Word-diff Side (plain text) against Their (resolved from Diff).
///
/// Returns `(side_ranges, diff_ranges)` — char-index ranges in the rope for
/// words that appear in Side but not in Their (shown on the Side section) and
/// words that appear in Their but not in Side (shown on the Diff section).
/// Both use the "added" (green) style.  The always-on intra-Diff word-diff
/// provides the `-`/`+` line-level highlighting within Diff sections.
pub fn refine_side_diff_added(
    text: &Rope,
    side_section: &Section,
    diff_section: &Section,
) -> (Vec<Range<usize>>, Vec<Range<usize>>) {
    let side_str = text
        .slice(side_section.content_start..side_section.content_end)
        .to_string();
    let (their_str, their_map) = resolve_diff_content_mapped(text, diff_section);

    let left_tokens = StringWordTokens::new(&side_str);
    let right_tokens = StringWordTokens::new(&their_str);
    let left_positions = left_tokens.positions.clone();
    let right_positions = right_tokens.positions.clone();
    let input = InternedInput::new(left_tokens, right_tokens);
    let diff = Diff::compute(Algorithm::Histogram, &input);

    let mut side_ranges: Vec<Range<usize>> = Vec::new();
    let mut diff_ranges: Vec<Range<usize>> = Vec::new();

    for hunk in diff.hunks() {
        // Words in Side (left) not in Their → show on Side section (green)
        for &(s, e) in &left_positions[hunk.before.start as usize..hunk.before.end as usize] {
            let rope_start = side_section.content_start + side_str[..s].chars().count();
            let rope_end = side_section.content_start + side_str[..e].chars().count();
            match side_ranges.last_mut() {
                Some(r) if r.end == rope_start => r.end = rope_end,
                _ => side_ranges.push(rope_start..rope_end),
            }
        }
        // Words in Their (right) not in Side → show on Diff section (green)
        for &(s, e) in &right_positions[hunk.after.start as usize..hunk.after.end as usize] {
            let char_s = their_str[..s].chars().count();
            let char_e = their_str[..e].chars().count();
            let rope_start = their_map[char_s];
            let rope_end = if char_e < their_map.len() {
                their_map[char_e]
            } else {
                rope_start + their_str[s..e].chars().count()
            };
            match diff_ranges.last_mut() {
                Some(r) if r.end == rope_start => r.end = rope_end,
                _ => diff_ranges.push(rope_start..rope_end),
            }
        }
    }

    (side_ranges, diff_ranges)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ropey::Rope;

    fn rope(s: &str) -> Rope {
        Rope::from(s)
    }

    fn parse_one(s: &str) -> (Rope, ConflictRegion) {
        let r = Rope::from(s);
        let mut c = find_conflicts(&r);
        assert_eq!(c.len(), 1, "expected exactly one conflict");
        (r, c.swap_remove(0))
    }

    // ── find_conflicts: git format ────────────────────────────────────────────

    #[test]
    fn no_conflicts() {
        assert!(find_conflicts(&rope("hello\nworld\n")).is_empty());
    }

    #[test]
    fn bare_conflict() {
        // A manually-edited conflict with no inner markers — just <<<<<<< / content / >>>>>>>
        let (r, c) = parse_one("<<<<<<< HEAD\nhand-picked\n>>>>>>> branch\n");
        assert_eq!(c.sections.len(), 1);
        assert_eq!(c.sections[0].kind, SectionKind::Side);
        let (s, e) = current_content(&c);
        assert_eq!(r.slice(s..e).to_string(), "hand-picked\n");
        let (s, e) = incoming_content(&c);
        assert_eq!(r.slice(s..e).to_string(), "hand-picked\n");
        assert!(base_content(&c).is_none());
    }

    #[test]
    fn two_way_conflict_basic() {
        let (r, c) = parse_one("<<<<<<< HEAD\ncurrent\n=======\nincoming\n>>>>>>> branch\n");
        assert_eq!(c.sections.len(), 2);
        assert_eq!(c.sections[0].kind, SectionKind::Side);
        assert_eq!(c.sections[1].kind, SectionKind::Side);
        assert_eq!(c.start, 0);
        assert_eq!(c.end, r.len_chars());
    }

    #[test]
    fn two_way_conflict_content() {
        let (r, c) =
            parse_one("<<<<<<< HEAD\ncurrent line\n=======\nincoming line\n>>>>>>> branch\n");
        let (s, e) = current_content(&c);
        assert_eq!(r.slice(s..e).to_string(), "current line\n");
        let (s, e) = incoming_content(&c);
        assert_eq!(r.slice(s..e).to_string(), "incoming line\n");
    }

    #[test]
    fn three_way_conflict_basic() {
        let (_, c) = parse_one(
            "<<<<<<< HEAD\ncurrent\n||||||| base\noriginal\n=======\nincoming\n>>>>>>> branch\n",
        );
        assert_eq!(c.sections.len(), 3);
        assert_eq!(c.sections[0].kind, SectionKind::Side);
        assert_eq!(c.sections[1].kind, SectionKind::Base);
        assert_eq!(c.sections[2].kind, SectionKind::Side);
    }

    #[test]
    fn three_way_content() {
        let text =
            "<<<<<<< HEAD\ncurrent\n||||||| base\nbase line\n=======\nincoming\n>>>>>>> branch\n";
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let (s, e) = current_content(c);
        assert_eq!(r.slice(s..e).to_string(), "current\n");
        let (s, e) = base_content(c).unwrap();
        assert_eq!(r.slice(s..e).to_string(), "base line\n");
        let (s, e) = incoming_content(c);
        assert_eq!(r.slice(s..e).to_string(), "incoming\n");
    }

    #[test]
    fn multiple_conflicts() {
        let text = concat!(
            "before\n",
            "<<<<<<< HEAD\na\n=======\nb\n>>>>>>> b\n",
            "between\n",
            "<<<<<<< HEAD\nc\n=======\nd\n>>>>>>> b\n",
            "after\n",
        );
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        assert_eq!(conflicts.len(), 2);
        assert!(conflicts[1].start > conflicts[0].end);
    }

    #[test]
    fn conflict_no_trailing_newline() {
        let text = "<<<<<<< HEAD\ncurrent\n=======\nincoming\n>>>>>>> branch";
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].end, text.chars().count());
    }

    #[test]
    fn incomplete_no_close_ignored() {
        assert!(find_conflicts(&rope("<<<<<<< HEAD\ncurrent\n=======\nincoming\n")).is_empty());
    }

    #[test]
    fn restarted_on_nested_open_marker() {
        let text = concat!(
            "<<<<<<< outer\n",
            "<<<<<<< inner\n",
            "current\n",
            "=======\n",
            "incoming\n",
            ">>>>>>> inner\n",
        );
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].start, "<<<<<<< outer\n".chars().count());
    }

    #[test]
    fn jj_style_git_diff3_labels() {
        // jj produces diff3 markers with change IDs
        let text = concat!(
            "<<<<<<< ouyysnvk c9a24f82 \"first version\"\n",
            "1st version\n",
            "||||||| zxwrknxy 62f152a0 \"base\"\n",
            "original\n",
            "=======\n",
            "2nd version\n",
            ">>>>>>> kyqztmxm cf165681 \"second version\"\n",
        );
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        assert_eq!(conflicts.len(), 1);
        let c = &conflicts[0];
        assert_eq!(c.sections.len(), 3);
        let (s, e) = current_content(c);
        assert_eq!(r.slice(s..e).to_string(), "1st version\n");
        let (s, e) = base_content(c).unwrap();
        assert_eq!(r.slice(s..e).to_string(), "original\n");
        let (s, e) = incoming_content(c);
        assert_eq!(r.slice(s..e).to_string(), "2nd version\n");
    }

    #[test]
    fn jj_snapshot_two_sides() {
        // jj snapshot format: 2 sides + 1 base
        let (r, c) = parse_one(concat!(
            "<<<<<<< Conflict 1 of 1\n",
            "+++++++ side #1\n",
            "alpha\n",
            "------- base\n",
            "beta\n",
            "+++++++ side #2\n",
            "gamma\n",
            ">>>>>>> Conflict 1 of 1 ends\n",
        ));
        assert_eq!(c.sections.len(), 3);
        assert_eq!(c.sections[0].kind, SectionKind::Side);
        assert_eq!(c.sections[1].kind, SectionKind::Base);
        assert_eq!(c.sections[2].kind, SectionKind::Side);
        let (s, e) = current_content(&c);
        assert_eq!(r.slice(s..e).to_string(), "alpha\n");
        let (s, e) = base_content(&c).unwrap();
        assert_eq!(r.slice(s..e).to_string(), "beta\n");
        let (s, e) = incoming_content(&c);
        assert_eq!(r.slice(s..e).to_string(), "gamma\n");
    }

    #[test]
    fn refine_pairs_jj_snapshot_three_sides() {
        // 5 sections [S0,B1,S2,B3,S4]
        // Phase 1: base-last-side (3,4)
        // Phase 2: side-side     (0,2), (0,4), (2,4) — C(3,2)
        let text = concat!(
            "<<<<<<< Conflict 1 of 1\n",
            "+++++++ side #1\n",
            "s1\n",
            "------- base\n",
            "base\n",
            "+++++++ side #2\n",
            "s2\n",
            "------- base\n",
            "base\n",
            "+++++++ side #3\n",
            "s3\n",
            ">>>>>>> Conflict 1 of 1 ends\n",
        );
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        assert_eq!(conflicts.len(), 1);
        let c = &conflicts[0];
        assert_eq!(c.sections.len(), 5);
        assert_eq!(c.sections[0].kind, SectionKind::Side);
        assert_eq!(c.sections[1].kind, SectionKind::Base);
        assert_eq!(c.sections[2].kind, SectionKind::Side);
        assert_eq!(c.sections[3].kind, SectionKind::Base);
        assert_eq!(c.sections[4].kind, SectionKind::Side);
        assert_eq!(c.num_refine_pairs(), 4);
    }

    #[test]
    fn jj_diff_format() {
        let text = concat!(
            "<<<<<<< conflict 1 of 1\n",
            "%%%%%%% diff from: vpxusssl 38d49363 \"merge base\"\n",
            "\\\\\\\\\\\\\\        to: rtsqusxu 2768b0b9 \"commit A\"\n",
            " apple\n",
            "-grape\n",
            "+grapefruit\n",
            " orange\n",
            "+++++++ ysrnknol 7a20f389 \"commit B\"\n",
            "APPLE\n",
            "GRAPE\n",
            "ORANGE\n",
            ">>>>>>> conflict 1 of 1 ends\n",
        );
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        assert_eq!(conflicts.len(), 1);
        let c = &conflicts[0];
        assert_eq!(c.sections.len(), 2);
        assert_eq!(c.sections[0].kind, SectionKind::Diff);
        assert_eq!(c.sections[0].marker_lines, 2);
        assert_eq!(
            r.slice(c.sections[0].content_start..c.sections[0].content_end)
                .to_string(),
            " apple\n-grape\n+grapefruit\n orange\n"
        );
        assert_eq!(
            r.slice(c.sections[0].marker_start..c.sections[0].content_start)
                .to_string(),
            "%%%%%%% diff from: vpxusssl 38d49363 \"merge base\"\n\\\\\\\\\\\\\\        to: rtsqusxu 2768b0b9 \"commit A\"\n"
        );
        assert_eq!(c.sections[1].kind, SectionKind::Side);
        assert_eq!(c.sections[1].marker_lines, 1);
    }

    #[test]
    fn jj_snapshot_section_content() {
        let text = concat!(
            "<<<<<<< Conflict\n",
            "+++++++ s1\n",
            "hello\n",
            "------- base\n",
            "world\n",
            "+++++++ s2\n",
            "rust\n",
            ">>>>>>> Conflict ends\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let (s, e) = (c.sections[0].content_start, c.sections[0].content_end);
        assert_eq!(r.slice(s..e).to_string(), "hello\n");
        let (s, e) = (c.sections[1].content_start, c.sections[1].content_end);
        assert_eq!(r.slice(s..e).to_string(), "world\n");
        let (s, e) = (c.sections[2].content_start, c.sections[2].content_end);
        assert_eq!(r.slice(s..e).to_string(), "rust\n");
    }
    // ── conflict_at ───────────────────────────────────────────────────────────

    #[test]
    fn conflict_at_inside() {
        let text = "before\n<<<<<<< HEAD\ncurrent\n=======\nincoming\n>>>>>>> b\nafter\n";
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        let before_len = "before\n".chars().count();
        assert_eq!(conflict_at(&conflicts, before_len), Some(0));
        assert_eq!(conflict_at(&conflicts, 0), None);
        assert_eq!(conflict_at(&conflicts, conflicts[0].end + 1), None);

        // Boundary: at start (inclusive) and at end (exclusive)
        assert_eq!(conflict_at(&conflicts, conflicts[0].start), Some(0));
        assert_eq!(conflict_at(&conflicts, conflicts[0].end), None);
    }

    // ── next/prev_conflict ────────────────────────────────────────────────────

    #[test]
    fn next_conflict_from_before() {
        let text = concat!(
            "before\n",
            "<<<<<<< HEAD\na\n=======\nb\n>>>>>>> b\n",
            "between\n",
            "<<<<<<< HEAD\nc\n=======\nd\n>>>>>>> b\n",
        );
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        assert_eq!(next_conflict(&conflicts, 0), Some(0));
        assert_eq!(next_conflict(&conflicts, conflicts[0].start + 1), Some(1));
        assert_eq!(next_conflict(&conflicts, conflicts[1].end), None);
    }

    #[test]
    fn prev_conflict_from_after() {
        let text = concat!(
            "<<<<<<< HEAD\na\n=======\nb\n>>>>>>> b\n",
            "between\n",
            "<<<<<<< HEAD\nc\n=======\nd\n>>>>>>> b\n",
            "after\n",
        );
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        let after_last = conflicts[1].end + 1;
        assert_eq!(prev_conflict(&conflicts, after_last), Some(1));
        let inside_second = conflicts[1].start + 1;
        assert_eq!(prev_conflict(&conflicts, inside_second), Some(1));
        assert_eq!(prev_conflict(&conflicts, conflicts[1].start), Some(0));
        assert_eq!(prev_conflict(&conflicts, 0), None);
    }

    // ── conflict_section_at ───────────────────────────────────────────────────

    #[test]
    fn section_at_git_two_way() {
        let text = "<<<<<<< HEAD\ncurrent\n=======\nincoming\n>>>>>>> b\n";
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        // On <<<<<<< line → section 0
        assert_eq!(conflict_section_at(c, &r, 0), Some(0));
        // In current content → section 0
        let current_pos = "<<<<<<< HEAD\n".chars().count();
        assert_eq!(conflict_section_at(c, &r, current_pos), Some(0));
        // On ======= line → None
        let sep_pos = "<<<<<<< HEAD\ncurrent\n".chars().count();
        assert_eq!(conflict_section_at(c, &r, sep_pos), None);
        // In incoming content → section 1
        let incoming_pos = "<<<<<<< HEAD\ncurrent\n=======\n".chars().count();
        assert_eq!(conflict_section_at(c, &r, incoming_pos), Some(1));
        // On >>>>>>> line → section 1
        let close_pos = "<<<<<<< HEAD\ncurrent\n=======\nincoming\n".chars().count();
        assert_eq!(conflict_section_at(c, &r, close_pos), Some(1));
    }

    #[test]
    fn section_at_git_three_way() {
        let text = "<<<<<<< HEAD\ncurrent\n||||||| base\nbase\n=======\nincoming\n>>>>>>> b\n";
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        // On ||||||| line → section 1
        let base_marker = "<<<<<<< HEAD\ncurrent\n".chars().count();
        assert_eq!(conflict_section_at(c, &r, base_marker), Some(1));
        // In base content → section 1
        let base_pos = "<<<<<<< HEAD\ncurrent\n||||||| base\n".chars().count();
        assert_eq!(conflict_section_at(c, &r, base_pos), Some(1));
        // On ======= → None
        let sep_pos = "<<<<<<< HEAD\ncurrent\n||||||| base\nbase\n"
            .chars()
            .count();
        assert_eq!(conflict_section_at(c, &r, sep_pos), None);
    }

    #[test]
    fn section_at_jj_snapshot() {
        let text = concat!(
            "<<<<<<< Conflict\n",
            "+++++++ s1\n",
            "hello\n",
            "------- base\n",
            "world\n",
            "+++++++ s2\n",
            "rust\n",
            ">>>>>>> Conflict ends\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        // On +++++++ s1 line → section 0
        let s1_marker = "<<<<<<< Conflict\n".chars().count();
        assert_eq!(conflict_section_at(c, &r, s1_marker), Some(0));
        // On ------- line → section 1
        let base_marker = "<<<<<<< Conflict\n+++++++ s1\nhello\n".chars().count();
        assert_eq!(conflict_section_at(c, &r, base_marker), Some(1));
        // On +++++++ s2 line → section 2
        let s2_marker = "<<<<<<< Conflict\n+++++++ s1\nhello\n------- base\nworld\n"
            .chars()
            .count();
        assert_eq!(conflict_section_at(c, &r, s2_marker), Some(2));
        // On >>>>>>> line → section 2 (last)
        let close = "<<<<<<< Conflict\n+++++++ s1\nhello\n------- base\nworld\n+++++++ s2\nrust\n"
            .chars()
            .count();
        assert_eq!(conflict_section_at(c, &r, close), Some(2));
    }

    // ── refine pairs ──────────────────────────────────────────────────────────

    type RefinePairCase<'a> = (&'a str, usize, &'a [(usize, usize)]);

    #[test]
    fn refine_pairs_cases() {
        let cases: &[RefinePairCase<'_>] = &[
            (
                "<<<<<<< HEAD\ncurrent\n=======\nincoming\n>>>>>>> b\n",
                1,
                &[(0, 1)],
            ),
            (
                "<<<<<<< HEAD\ncurrent\n||||||| base\nbase\n=======\nincoming\n>>>>>>> b\n",
                2,
                &[(1, 2), (0, 2)], // Phase 1: base-side; Phase 2: side-side
            ),
        ];
        for &(text, num_pairs, pairs) in cases {
            let c = &find_conflicts(&rope(text))[0];
            assert_eq!(c.num_refine_pairs(), num_pairs);
            for (i, &expected) in pairs.iter().enumerate() {
                assert_eq!(c.refine_pair_indices(i), Some(expected));
            }
            assert_eq!(c.refine_pair_indices(pairs.len()), None);
        }
    }

    #[test]
    fn refine_pairs_three_way() {
        // 3 sections [S0,B1,S2]
        // Phase 1: base-last-side (1,2)
        // Phase 2: side-side      (0,2)
        let text = "<<<<<<< HEAD\ncurrent\n||||||| base\nbase\n=======\nincoming\n>>>>>>> b\n";
        let c = &find_conflicts(&rope(text))[0];
        assert_eq!(c.num_refine_pairs(), 2);
        assert_eq!(c.refine_pair_indices(0), Some((1, 2)));
        assert_eq!(c.refine_pair_indices(1), Some((0, 2)));
        assert_eq!(c.refine_pair_indices(2), None);
    }

    #[test]
    fn refine_pairs_four_sections() {
        // 4 sections [S0,B1,S2,S3]
        // Phase 1: base-last-side  (1,3)
        // Phase 2: side-side      (0,2), (0,3), (2,3)
        let text = concat!(
            "<<<<<<< Conflict\n",
            "+++++++ s1\nA\n",
            "------- base\nB\n",
            "+++++++ s2\nC\n",
            "+++++++ s3\nD\n",
            ">>>>>>> Conflict ends\n",
        );
        let c = &find_conflicts(&rope(text))[0];
        assert_eq!(c.sections.len(), 4);
        assert_eq!(c.num_refine_pairs(), 4);
        assert_eq!(c.refine_pair_indices(0), Some((1, 3)));
        assert_eq!(c.refine_pair_indices(1), Some((0, 2)));
        assert_eq!(c.refine_pair_indices(2), Some((0, 3)));
        assert_eq!(c.refine_pair_indices(3), Some((2, 3)));
        assert_eq!(c.refine_pair_indices(4), None);
    }

    #[test]
    fn pair_sections_no_base_swap() {
        // diff3: sections = [Side(current), Base, Side(incoming)]
        // Pair 0: (1,2) = (Base, Side) → no swap (not Diff+Side)
        // Pair 1: (0,2) = (Side, Side) → no swap, kept in order
        let text = "<<<<<<< HEAD\ncurrent\n||||||| base\nbase\n=======\nincoming\n>>>>>>> b\n";
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        assert_eq!(c.num_refine_pairs(), 2);

        // pair 0 => (1,2) = (Base, Side) → no swap
        let (left, right) = conflict_pair_sections(c, 0).unwrap();
        assert_eq!(r.slice(left.0..left.1).to_string(), "base\n");
        assert_eq!(r.slice(right.0..right.1).to_string(), "incoming\n");

        // pair 1 => (0,2) = (Side, Side) → no swap, kept in order
        let (left, right) = conflict_pair_sections(c, 1).unwrap();
        assert_eq!(r.slice(left.0..left.1).to_string(), "current\n");
        assert_eq!(r.slice(right.0..right.1).to_string(), "incoming\n");

        assert!(conflict_pair_sections(c, 2).is_none());
    }
    #[test]
    fn refine_pair_clamped_and_default() {
        let text = "<<<<<<< HEAD\ncurrent\n=======\nincoming\n>>>>>>> b\n"; // 1 pair, max idx 0
        let r = rope(text);
        let c = &find_conflicts(&r)[0];

        // Stale/out-of-range selection gets clamped to valid range
        let mut state = HashMap::new();
        state.insert(c.start, 99);
        assert_eq!(conflict_refine_pair(&state, c), 0);

        // Empty state defaults to 0
        assert_eq!(conflict_refine_pair(&HashMap::new(), c), 0);

        // Valid index passes through
        state.clear();
        state.insert(c.start, 0);
        assert_eq!(conflict_refine_pair(&state, c), 0);
    }

    #[test]
    fn conflict_marker_lines_returns_sorted() {
        let text = concat!(
            "before\n",
            "<<<<<<< HEAD\ncurrent\n||||||| base\nbase\n=======\nincoming\n>>>>>>> b\n",
            "after\n",
        );
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        let lines = conflict_marker_lines(&conflicts, &r);
        assert!(!lines.is_empty());
        for i in 0..lines.len() - 1 {
            assert!(lines[i] < lines[i + 1]);
        }
        let dedup_check: Vec<_> = lines.iter().collect();
        assert_eq!(dedup_check.len(), lines.len());
    }

    #[test]
    fn all_sides_content_concatenates() {
        let text = concat!(
            "<<<<<<< Conflict\n",
            "+++++++ s1\nfirst\n",
            "------- base\nbase\n",
            "+++++++ s2\nsecond\n",
            "+++++++ s3\nthird\n",
            ">>>>>>> Conflict ends\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let all = all_sides_content(&r, c);
        assert!(all.contains("first"));
        assert!(all.contains("second"));
        assert!(all.contains("third"));
        // Base should NOT be included
        assert!(!all.contains("base\n"));
    }

    #[test]
    fn empty_side_conflict() {
        let text = "<<<<<<<\n=======\n>>>>>>>\n";
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        assert_eq!(conflicts.len(), 1);
        let c = &conflicts[0];
        assert_eq!(c.sections.len(), 2);
        let (s, e) = current_content(c);
        assert_eq!(r.slice(s..e).to_string(), "");
        let (s, e) = incoming_content(c);
        assert_eq!(r.slice(s..e).to_string(), "");
    }

    #[test]
    fn conflict_with_crlf() {
        let text = "<<<<<<< HEAD\r\ncurrent\r\n=======\r\nincoming\r\n>>>>>>> b\r\n";
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        assert_eq!(conflicts.len(), 1);
    }

    #[test]
    fn transition_to_jj_with_content_section() {
        // Git→Jj transition where the implicit git side has content
        let text = concat!(
            "<<<<<<< Conflict\n",
            "shared content\n",
            "+++++++ side #1\n",
            "more\n",
            ">>>>>>> Conflict ends\n",
        );
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        assert_eq!(conflicts.len(), 1);
        let c = &conflicts[0];
        // The git side with "shared content" should be preserved
        assert_eq!(c.sections.len(), 2);
        let (s, e) = current_content(c);
        assert_eq!(r.slice(s..e).to_string(), "shared content\n");
        let (s, e) = incoming_content(c);
        assert_eq!(r.slice(s..e).to_string(), "more\n");
    }

    // ── refine_diff ───────────────────────────────────────────────────────────

    #[test]
    fn refine_diff_cases() {
        for &(current, incoming) in &[
            ("hello world", "hello world"),        // identical
            ("hello world", "hello rust"),         // single word
            ("foo bar", "baz qux"),                // completely different
            ("hello world test", "hello foo bar"), // adjacent words
        ] {
            let text = format!(
                "<<<<<<< HEAD\n{}\n=======\n{}\n>>>>>>> b\n",
                current, incoming
            );
            let r = Rope::from(text.as_str());
            let c = &find_conflicts(&r)[0];
            let (removed, added) = refine_diff(&r, current_content(c), incoming_content(c));

            if current == incoming {
                assert!(removed.is_empty());
                assert!(added.is_empty());
                continue;
            }

            assert!(
                !removed.is_empty(),
                "expected removed for '{}' vs '{}'",
                current,
                incoming
            );
            assert!(
                !added.is_empty(),
                "expected added for '{}' vs '{}'",
                current,
                incoming
            );

            let removed_text: String = removed
                .iter()
                .map(|range| r.slice(range.clone()).to_string())
                .collect::<Vec<_>>()
                .join(" ");
            let added_text: String = added
                .iter()
                .map(|range| r.slice(range.clone()).to_string())
                .collect::<Vec<_>>()
                .join(" ");

            for word in current.split_whitespace() {
                if !incoming.contains(word) {
                    assert!(
                        removed_text.contains(word),
                        "'{word}' missing from removed: '{removed_text}'"
                    );
                }
            }
            for word in incoming.split_whitespace() {
                if !current.contains(word) {
                    assert!(
                        added_text.contains(word),
                        "'{word}' missing from added: '{added_text}'"
                    );
                }
            }
        }
    }

    #[test]
    fn refine_diff_identical_sections() {
        // When both sides are identical, refine_diff should return no ranges.
        let text = "<<<<<<< HEAD\nhello world\n=======\nhello world\n>>>>>>> branch\n";
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        let left = current_content(&conflicts[0]);
        let right = incoming_content(&conflicts[0]);
        let (removed, added) = refine_diff(&r, left, right);
        assert!(
            removed.is_empty(),
            "expected no removed ranges, got {:?}",
            removed
        );
        assert!(
            added.is_empty(),
            "expected no added ranges, got {:?}",
            added
        );
    }

    #[test]
    fn refine_diff_single_word_change() {
        // "hello world" vs "hello rust" — only "world"/"rust" differ.
        let text = "<<<<<<< HEAD\nhello world\n=======\nhello rust\n>>>>>>> branch\n";
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        let left = current_content(&conflicts[0]);
        let right = incoming_content(&conflicts[0]);
        let (removed, added) = refine_diff(&r, left, right);

        assert_eq!(
            removed.len(),
            1,
            "expected 1 removed range, got {:?}",
            removed
        );
        assert_eq!(added.len(), 1, "expected 1 added range, got {:?}", added);

        let removed_word: String = r.slice(removed[0].clone()).chars().collect();
        let added_word: String = r.slice(added[0].clone()).chars().collect();
        assert_eq!(removed_word, "world");
        assert_eq!(added_word, "rust");
    }

    #[test]
    fn refine_diff_completely_different() {
        // No words in common — all tokens should be flagged.
        let text = "<<<<<<< HEAD\nfoo bar\n=======\nbaz qux\n>>>>>>> branch\n";
        let r = rope(text);
        let conflicts = find_conflicts(&r);
        let left = current_content(&conflicts[0]);
        let right = incoming_content(&conflicts[0]);
        let (removed, added) = refine_diff(&r, left, right);

        // Both "foo" and "bar" removed; "baz" and "qux" added (may be merged).
        assert!(!removed.is_empty(), "expected removed ranges");
        assert!(!added.is_empty(), "expected added ranges");

        let removed_text: String = removed
            .iter()
            .map(|range| r.slice(range.clone()).to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let added_text: String = added
            .iter()
            .map(|range| r.slice(range.clone()).to_string())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            removed_text.contains("foo"),
            "expected 'foo' in removed: {removed_text}"
        );
        assert!(
            removed_text.contains("bar"),
            "expected 'bar' in removed: {removed_text}"
        );
        assert!(
            added_text.contains("baz"),
            "expected 'baz' in added: {added_text}"
        );
        assert!(
            added_text.contains("qux"),
            "expected 'qux' in added: {added_text}"
        );
    }

    #[test]
    fn refine_diff_current_incoming_ws() {
        // Regression: whitespace between None and } in current-incoming
        // should highlight only the 4 extra spaces (common \n is skipped).
        let text = concat!(
            "<<<<<<< side #1\n",
            "    let location =\n",
            "        if args.onto.is_none() && args.insert_after.is_none() ",
            "&& args.insert_before.is_none() {\n",
            "            None\n",
            "        } else {\n",
            "            Some(compute_commit_location(\n",
            "                ui,\n",
            "                &workspace_command,\n",
            "                args.onto.as_deref(),\n",
            "                args.insert_after.as_deref(),\n",
            "                args.insert_before.as_deref(),\n",
            "                \"duplicated commits\",\n",
            "            )?)\n",
            "        };\n",
            "||||||| base\n",
            "    let location = if args.destination.is_none()\n",
            "        && args.insert_after.is_none()\n",
            "        && args.insert_before.is_none()\n",
            "    {\n",
            "        None\n",
            "    } else {\n",
            "        Some(compute_commit_location(\n",
            "            ui,\n",
            "            &workspace_command,\n",
            "            args.destination.as_deref(),\n",
            "            args.insert_after.as_deref(),\n",
            "            args.insert_before.as_deref(),\n",
            "            \"duplicated commits\",\n",
            "        )?)\n",
            "    };\n",
            "=======\n",
            "    let location = if args.destination.is_none()\n",
            "        && args.insert_after.is_none()\n",
            "        && args.insert_before.is_none()\n",
            "    {\n",
            "        None\n",
            "    } else {\n",
            "        Some(compute_commit_location(\n",
            "            ui,\n",
            "            &workspace_command,\n",
            "            args.destination.as_deref(),\n",
            "            args.insert_after.as_deref(),\n",
            "            args.insert_before.as_deref(),\n",
            "            \"duplicated revisions\",\n",
            "        )?)\n",
            "    };\n",
            ">>>>>>> side #2\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let base = base_content(c).unwrap();
        let current = current_content(c);
        let incoming = incoming_content(c);

        // ── current-base ──────────────────────────────────────────────────
        let (removed, added) = refine_diff(&r, current, base);
        let pos_cur = r
            .to_string()
            .find("None\n        ")
            .expect("current-side None followed by 8 spaces");
        assert!(!removed.is_empty(), "current-base: expected removed ranges");
        assert!(!added.is_empty(), "current-base: expected added ranges");
        // Same None→} whitespace diff as current-incoming (4 extra spaces).
        // Match any 4-char removed range inside the "\n        }" gap.
        assert!(
            removed.iter().any(|range| {
                range.start >= pos_cur + 5
                    && range.end <= pos_cur + 13
                    && range.end - range.start == 4
            }),
            "current-base: expected 4 extra spaces between None and }}"
        );

        // ── base-incoming ─────────────────────────────────────────────────
        let (removed, added) = refine_diff(&r, base, incoming);
        assert!(
            !removed.is_empty(),
            "base-incoming: expected removed ranges"
        );
        assert!(!added.is_empty(), "base-incoming: expected added ranges");
        // Only the string differs — no whitespace changes.
        let rem_text: String = removed
            .iter()
            .map(|range| r.slice(range.clone()).to_string())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            !rem_text.contains("None"),
            "base-incoming: None should not be in removed: {rem_text:?}"
        );

        // ── current-incoming ──────────────────────────────────────────────
        let (removed, added) = refine_diff(&r, current, incoming);
        assert!(
            !removed.is_empty(),
            "current-incoming: expected removed ranges"
        );
        assert!(!added.is_empty(), "current-incoming: expected added ranges");
        assert!(
            removed.iter().any(|range| {
                range.start >= pos_cur + 5
                    && range.end <= pos_cur + 13
                    && range.end - range.start == 4
            }),
            "current-incoming: expected 4 extra spaces between None and }}"
        );
    }

    #[test]
    fn refine_diff_underscore_word() {
        // jj treats underscore and non-ASCII as word chars.
        // "fn foo_bar() {}" → "fn foo_baz() {}" should highlight only
        // "bar" vs "baz" — the fn, foo_, (){} should all match.
        let text = "<<<<<<< HEAD\nfn foo_bar() {}\n=======\nfn foo_baz() {}\n>>>>>>> branch\n";
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let (removed, added) = refine_diff(&r, current_content(c), incoming_content(c));
        let rem_text: String = removed
            .iter()
            .map(|range| r.slice(range.clone()).to_string())
            .collect::<Vec<_>>()
            .join("");
        let add_text: String = added
            .iter()
            .map(|range| r.slice(range.clone()).to_string())
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(
            removed.len(),
            1,
            "expected 1 removed range, got {removed:?}"
        );
        assert_eq!(added.len(), 1, "expected 1 added range, got {added:?}");
        // Underscore splits words, so only "bar" / "baz" are highlighted.
        assert_eq!(rem_text, "bar");
        assert_eq!(add_text, "baz");
    }

    #[test]
    fn refine_diff_line_level_match() {
        // Shared lines should be matched at the line level, not shown as changes.
        let text = concat!(
            "<<<<<<< HEAD\n",
            "shared_prefix\n",
            "only_in_current\n",
            "shared_suffix\n",
            "=======\n",
            "shared_prefix\n",
            "only_in_incoming\n",
            "shared_suffix\n",
            ">>>>>>> branch\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let (removed, added) = refine_diff(&r, current_content(c), incoming_content(c));
        let rem_text: String = removed
            .iter()
            .map(|range| r.slice(range.clone()).to_string())
            .collect::<Vec<_>>()
            .join("");
        let add_text: String = added
            .iter()
            .map(|range| r.slice(range.clone()).to_string())
            .collect::<Vec<_>>()
            .join("");
        // Only the differing word within the differing line should be
        // highlighted — "current" vs "incoming".  The rest is matched
        // at line → word → nonword levels.
        assert_eq!(rem_text, "current");
        assert_eq!(add_text, "incoming");
    }

    // ── refine_diff_section ────────────────────────────────────────────────────

    #[test]
    fn refine_diff_section_identical() {
        let text = concat!(
            "<<<<<<<\n",
            "+++++++ side\n",
            "content\n",
            "%%%%%%%\n",
            "\\\\\\\\\\\\\\\n",
            " same\n",
            " unchanged\n",
            ">>>>>>>\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let section = &c.sections[1];
        assert_eq!(section.kind, SectionKind::Diff);
        let (removed, added) = refine_diff_section(&r, section);
        assert!(removed.is_empty());
        assert!(added.is_empty());
    }

    #[test]
    fn refine_diff_section_single_word_change() {
        let text = concat!(
            "<<<<<<<\n",
            "+++++++ side\n",
            "content\n",
            "%%%%%%%\n",
            "\\\\\\\\\\\\\\\n",
            "-            \"duplicated commits\",\n",
            "+            \"duplicated revisions\",\n",
            ">>>>>>>\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let section = &c.sections[1];
        assert_eq!(section.kind, SectionKind::Diff);
        let (removed, added) = refine_diff_section(&r, section);
        assert_eq!(removed.len(), 1);
        assert_eq!(added.len(), 1);
        assert!(r.slice(removed[0].clone()).to_string().contains("commits"));
        assert!(r.slice(added[0].clone()).to_string().contains("revisions"));
    }

    #[test]
    fn refine_diff_section_context_ignored() {
        // Context lines (` ` prefix) should not contribute to the diff.
        let text = concat!(
            "<<<<<<<\n",
            "+ side\n",
            "%%%%%%%\n",
            "\\\\\\\\\\\\\\\n",
            "-old\n",
            " context\n",
            "+new\n",
            ">>>>>>>\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let section = &c.sections[1];
        assert_eq!(section.kind, SectionKind::Diff);
        let (removed, added) = refine_diff_section(&r, section);
        assert_eq!(removed.len(), 1);
        assert_eq!(added.len(), 1);
        assert_eq!(r.slice(removed[0].clone()).to_string(), "old");
        assert_eq!(r.slice(added[0].clone()).to_string(), "new");
    }

    // ── resolve_diff_content / resolve_diff_content_base ────────────────────────

    #[test]
    fn test_resolve_diff_content_side() {
        let text = concat!(
            "<<<<<<<\n+++++++ side\ncontent\n%%%%%%%\n\\\\\\\\\\\\\\\n",
            " apple\n-grape\n+grapefruit\n orange\n",
            ">>>>>>>\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let section = &c.sections[1];
        assert_eq!(section.kind, SectionKind::Diff);
        let resolved = resolve_diff_content(&r, section);
        assert_eq!(resolved, "apple\ngrapefruit\norange\n");
    }

    #[test]
    fn test_resolve_diff_content_base() {
        let text = concat!(
            "<<<<<<<\n+++++++ side\ncontent\n%%%%%%%\n\\\\\\\\\\\\\\\n",
            "-old\n context\n+new\n",
            ">>>>>>>\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let section = &c.sections[1];
        assert_eq!(section.kind, SectionKind::Diff);
        let resolved = resolve_diff_content_base(&r, section);
        assert_eq!(resolved, "old\ncontext\n");
    }

    #[test]
    fn test_resolve_diff_content_all_sides() {
        // (Side, Diff) conflict: all_sides_content should resolve the Diff
        let text = concat!(
            "<<<<<<<\n+++++++ side\nside content\n%%%%%%%\n\\\\\\\\\\\\\\\n",
            "-old\n+new\n",
            ">>>>>>>\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let all = all_sides_content(&r, c);
        assert!(all.contains("side content"));
        assert!(!all.contains("-old"));
        assert!(all.contains("new"));
    }

    // ── refine_side_with_base ──────────────────────────────────────────────────

    #[test]
    fn refine_side_added_words_basic() {
        // (Side, Diff): Side = "hello world test", Diff shows base="hello world old"
        // Added words in Side: "test"
        let text = concat!(
            "<<<<<<<\n+++++++ side\nhello world test\n%%%%%%%\n\\\\\\\\\\\\\\\n",
            "-hello world old\n+hello world new\n",
            ">>>>>>>\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let side = &c.sections[0];
        assert_eq!(side.kind, SectionKind::Side);
        let diff = &c.sections[1];
        let base = resolve_diff_content_base(&r, diff);
        let added = refine_side_with_base(&r, side, &base);
        // "test" should be highlighted as added
        assert!(!added.is_empty(), "expected added words");
        for range in &added {
            let word = r.slice(range.clone()).to_string();
            assert!(word == "test", "unexpected added word: {word:?}");
        }
    }

    #[test]
    fn refine_side_no_added_words() {
        // Side content matches the base (no change → no added words)
        let text = concat!(
            "<<<<<<<\n+++++++ side\nhello world\n%%%%%%%\n\\\\\\\\\\\\\\\n",
            " hello world\n",
            ">>>>>>>\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let side = &c.sections[0];
        let diff = &c.sections[1];
        let base = resolve_diff_content_base(&r, diff);
        assert_eq!(base, "hello world\n");
        let added = refine_side_with_base(&r, side, &base);
        assert!(added.is_empty(), "expected no added words, got {added:?}");
    }

    #[test]
    fn refine_side_all_words_added() {
        // Everything in Side is new vs base
        let text = concat!(
            "<<<<<<<\n+++++++ side\nbrand new content\n%%%%%%%\n\\\\\\\\\\\\\\\n",
            "-no\n+yes\n",
            ">>>>>>>\n",
        );
        let r = rope(text);
        let c = &find_conflicts(&r)[0];
        let side = &c.sections[0];
        let diff = &c.sections[1];
        let base = resolve_diff_content_base(&r, diff);
        let added = refine_side_with_base(&r, side, &base);
        // All three words are new
        assert_eq!(added.len(), 3, "expected 3 added words");
        let words: Vec<String> = added
            .iter()
            .map(|range| r.slice(range.clone()).to_string())
            .collect();
        assert_eq!(words, vec!["brand", "new", "content"]);
    }
}
