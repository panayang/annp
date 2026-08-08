//! Real text: a SentencePiece BPE tokenizer and a streaming corpus reader.
//!
//! Everything in this file is vendored rather than pulled from a crate, for the
//! same reason `rng.rs` is: a measurement that cannot be reproduced years later
//! from the repository alone is not a measurement. A tokenizer is worse than
//! most dependencies in that respect, because a silently different segmentation
//! changes every number downstream without ever failing.
//!
//! Correctness here is not argued, it is pinned. `tests` holds token id
//! sequences produced by the reference `sentencepiece` implementation for the
//! model in the working directory, and the encoder must reproduce them exactly.
//!
//! What the model in hand actually specifies, read out of its own protobuf
//! rather than assumed:
//!
//! * `model_type = BPE` — scores are negated merge ranks, not log probabilities,
//!   so inference is greedy pair merging and *not* the Viterbi lattice a unigram
//!   model would need.
//! * `normalizer = identity`, with an empty `precompiled_charsmap`. No NFKC
//!   table to reimplement.
//! * `add_dummy_prefix = false` — no space is prepended, so `"The cat"` starts
//!   with the piece `The`, not `▁The`.
//! * `escape_whitespaces = true` (absent, and its default is true) — a space
//!   becomes U+2581 before segmentation.
//! * 256 pieces of type `BYTE`, so anything unsegmentable falls back to bytes
//!   and the encoder never has to emit `<unk>` for ordinary text.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};

use std::path::Path;

/// U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's stand-in for a space.
const SPACE: char = '\u{2581}';

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
    Unused,
}

impl PieceKind {
    fn from_proto(v: u64) -> Self {
        match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

/// A SentencePiece BPE model, loaded from the on-disk protobuf.
pub struct Tokenizer {
    pieces: Vec<String>,
    /// Negated merge rank: a higher score merges earlier.
    scores: Vec<f32>,
    kinds: Vec<PieceKind>,
    index: HashMap<String, u32>,
    /// `<0x00>`..`<0xFF>`, so an unsegmentable character can still be encoded.
    byte_ids: Vec<u32>,
    unk_id: u32,
}

/// Minimal protobuf wire reader. Only the shapes this file needs.
struct Wire<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Wire<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }

    fn done(&self) -> bool {
        self.i >= self.b.len()
    }

    fn varint(&mut self) -> Option<u64> {
        let (mut out, mut shift) = (0u64, 0u32);
        loop {
            let byte = *self.b.get(self.i)?;
            self.i += 1;
            out |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Some(out);
            }
            shift += 7;
            if shift > 63 {
                return None;
            }
        }
    }

    fn bytes(&mut self) -> Option<&'a [u8]> {
        let len = self.varint()? as usize;
        let out = self.b.get(self.i..self.i + len)?;
        self.i += len;
        Some(out)
    }

    fn fixed32(&mut self) -> Option<f32> {
        let raw = self.b.get(self.i..self.i + 4)?;
        self.i += 4;
        Some(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    /// Advances past a field whose contents are not wanted.
    fn skip(&mut self, wire: u8) -> Option<()> {
        match wire {
            0 => self.varint().map(|_| ()),
            1 => {
                self.i += 8;
                Some(())
            }
            2 => self.bytes().map(|_| ()),
            5 => self.fixed32().map(|_| ()),
            _ => None,
        }
    }
}

impl Tokenizer {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let raw = std::fs::read(path)?;
        let (mut pieces, mut scores, mut kinds) = (Vec::new(), Vec::new(), Vec::new());
        let mut w = Wire::new(&raw);
        while !w.done() {
            let Some(key) = w.varint() else { break };
            let (field, wire) = ((key >> 3) as u32, (key & 7) as u8);
            // Field 1 of ModelProto is the repeated piece list; everything else
            // (trainer spec, normalizer spec) has already been read off this
            // file by hand and is recorded in the module comment.
            if field == 1 && wire == 2 {
                let Some(sub) = w.bytes() else { break };
                let mut s = Wire::new(sub);
                let (mut piece, mut score, mut kind) = (String::new(), 0.0f32, PieceKind::Normal);
                while !s.done() {
                    let Some(k) = s.varint() else { break };
                    let (f, t) = ((k >> 3) as u32, (k & 7) as u8);
                    match (f, t) {
                        (1, 2) => {
                            let Some(b) = s.bytes() else { break };
                            piece = String::from_utf8_lossy(b).into_owned();
                        }
                        (2, 5) => score = s.fixed32().unwrap_or(0.0),
                        (3, 0) => kind = PieceKind::from_proto(s.varint().unwrap_or(1)),
                        _ => {
                            if s.skip(t).is_none() {
                                break;
                            }
                        }
                    }
                }
                pieces.push(piece);
                scores.push(score);
                kinds.push(kind);
            } else if w.skip(wire).is_none() {
                break;
            }
        }
        assert!(!pieces.is_empty(), "no pieces found in {}", path.display());

        let mut index = HashMap::with_capacity(pieces.len());
        for (id, p) in pieces.iter().enumerate() {
            index.entry(p.clone()).or_insert(id as u32);
        }
        // Byte pieces are spelled `<0xAB>`; collect them in byte order so a
        // fallback is an index, not a string format at encode time.
        let mut byte_ids = vec![u32::MAX; 256];
        for (id, (p, k)) in pieces.iter().zip(&kinds).enumerate() {
            if *k == PieceKind::Byte
                && p.len() == 6
                && p.starts_with("<0x")
                && p.ends_with('>')
                && let Ok(v) = u8::from_str_radix(&p[3..5], 16)
            {
                byte_ids[v as usize] = id as u32;
            }
        }
        let unk_id = kinds
            .iter()
            .position(|k| *k == PieceKind::Unknown)
            .unwrap_or(0) as u32;
        Ok(Self {
            pieces,
            scores,
            kinds,
            index,
            byte_ids,
            unk_id,
        })
    }

    pub fn piece(&self, id: u32) -> &str {
        &self.pieces[id as usize]
    }

    /// Segments `text` and appends its token ids to `out`.
    ///
    /// Greedy BPE: start from single characters and repeatedly merge whichever
    /// adjacent pair forms the highest-scoring piece in the vocabulary, leftmost
    /// first on a tie. That tie rule is not cosmetic — it is what the reference
    /// implementation does, and the fixtures below fail without it.
    pub fn encode(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        // Identity normalisation, no dummy prefix; only whitespace escaping.
        let escaped: String = text
            .chars()
            .map(|c| if c == ' ' { SPACE } else { c })
            .collect();

        // Symbols as a doubly linked list over character boundaries, so a merge
        // is a pointer update rather than a shift of everything to its right.
        let chars: Vec<(usize, char)> = escaped.char_indices().collect();
        let n = chars.len();
        let mut start: Vec<usize> = Vec::with_capacity(n);
        let mut end: Vec<usize> = Vec::with_capacity(n);
        for (k, (off, c)) in chars.iter().enumerate() {
            start.push(*off);
            end.push(off + c.len_utf8());
            let _ = k;
        }
        let mut prev: Vec<isize> = (0..n as isize).map(|k| k - 1).collect();
        let mut next: Vec<isize> = (0..n as isize).map(|k| k + 1).collect();
        let mut alive = vec![true; n];

        // Max-heap on (score, leftmost). `Reverse` on the position makes a lower
        // index win a score tie.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        // The fourth element is the right symbol's extent as it stood when the
        // pair was queued. Without it a queued pair can fire after its right
        // half has grown: `next[l] == r` still holds, so a liveness check passes,
        // and the merge produces a string different from the one that was looked
        // up — which may not be a piece at all, and then the whole run falls back
        // to bytes. The reference implementation guards this with the same
        // recorded length.
        let mut heap: BinaryHeap<(ordered::F32, Reverse<usize>, usize, usize, usize)> =
            BinaryHeap::new();
        let push_pair = |heap: &mut BinaryHeap<_>, l: usize, r: usize, end: &Vec<usize>| {
            let joined = &escaped[start[l]..end[r]];
            if let Some(&id) = self.index.get(joined)
                && matches!(
                    self.kinds[id as usize],
                    PieceKind::Normal | PieceKind::UserDefined
                )
            {
                heap.push((
                    ordered::F32(self.scores[id as usize]),
                    Reverse(l),
                    l,
                    r,
                    end[r],
                ));
            }
        };
        for k in 0..n.saturating_sub(1) {
            push_pair(&mut heap, k, k + 1, &end);
        }
        while let Some((_, _, l, r, extent)) = heap.pop() {
            // Stale entry: an endpoint merged away, the pair stopped being
            // adjacent, or the right half grew after this was queued.
            if !alive[l] || !alive[r] || next[l] != r as isize || end[r] != extent {
                continue;
            }
            end[l] = end[r];
            alive[r] = false;
            let after = next[r];
            next[l] = after;
            if after < n as isize {
                prev[after as usize] = l as isize;
                push_pair(&mut heap, l, after as usize, &end);
            }
            let before = prev[l];
            if before >= 0 {
                push_pair(&mut heap, before as usize, l, &end);
            }
        }

        let mut k = 0usize;
        while k < n {
            if alive[k] {
                let sym = &escaped[start[k]..end[k]];
                match self.index.get(sym) {
                    Some(&id) => out.push(id),
                    // Byte fallback, so ordinary text never becomes `<unk>`.
                    None => {
                        for b in sym.as_bytes() {
                            let id = self.byte_ids[*b as usize];
                            out.push(if id == u32::MAX { self.unk_id } else { id });
                        }
                    }
                }
            }
            k += 1;
        }
    }
}

/// A total order on `f32` good enough for a merge-priority heap. The scores in
/// a BPE model are negated integer ranks, so NaN cannot occur; it is mapped to
/// the bottom rather than being allowed to panic.
mod ordered {
    #[derive(Clone, Copy, PartialEq)]
    pub struct F32(pub f32);
    impl Eq for F32 {}
    impl PartialOrd for F32 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for F32 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0
                .partial_cmp(&other.0)
                .unwrap_or(std::cmp::Ordering::Less)
        }
    }
}

/// Streams the `input_text` fields out of a JSON array of records without
/// holding the file in memory.
///
/// The corpus in hand is eleven megabytes, which would fit; the reason to
/// stream anyway is that the next one will not, and a loader that only works
/// below some size is a loader that has to be rewritten at the worst moment.
/// Memory here is one document plus a fixed buffer, whatever the file size.
pub struct Documents<R: BufRead> {
    reader: R,
    buf: Vec<u8>,
    at: usize,
}

impl Documents<BufReader<std::fs::File>> {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(Self {
            reader: BufReader::with_capacity(1 << 20, file),
            buf: Vec::new(),
            at: 0,
        })
    }
}

impl<R: BufRead> Documents<R> {
    /// Next byte, refilling from the reader as needed.
    fn next_byte(&mut self) -> Option<u8> {
        if self.at >= self.buf.len() {
            self.buf.clear();
            self.at = 0;
            let mut chunk = [0u8; 1 << 16];
            let read = std::io::Read::read(&mut self.reader, &mut chunk).ok()?;
            if read == 0 {
                return None;
            }
            self.buf.extend_from_slice(&chunk[..read]);
        }
        let b = self.buf[self.at];
        self.at += 1;
        Some(b)
    }

    /// Reads one JSON string body, assuming the opening quote is consumed.
    fn string_body(&mut self) -> Option<String> {
        let mut out = String::new();
        loop {
            let b = self.next_byte()?;
            match b {
                b'"' => return Some(out),
                b'\\' => {
                    let e = self.next_byte()?;
                    match e {
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let mut hex = String::new();
                            for _ in 0..4 {
                                hex.push(self.next_byte()? as char);
                            }
                            let code = u32::from_str_radix(&hex, 16).ok()?;
                            // Surrogate pair: the low half follows as its own
                            // escape, and dropping it would silently corrupt any
                            // character outside the basic plane.
                            if (0xD800..0xDC00).contains(&code) {
                                if self.next_byte()? == b'\\' && self.next_byte()? == b'u' {
                                    let mut low = String::new();
                                    for _ in 0..4 {
                                        low.push(self.next_byte()? as char);
                                    }
                                    let lo = u32::from_str_radix(&low, 16).ok()?;
                                    let joined = 0x10000 + ((code - 0xD800) << 10) + (lo - 0xDC00);
                                    out.push(char::from_u32(joined)?);
                                }
                            } else {
                                out.push(char::from_u32(code)?);
                            }
                        }
                        other => out.push(other as char),
                    }
                }
                _ => {
                    // UTF-8 continuation bytes pass through untouched.
                    let mut bytes = vec![b];
                    let extra = match b {
                        0x00..=0x7f => 0,
                        0xc0..=0xdf => 1,
                        0xe0..=0xef => 2,
                        _ => 3,
                    };
                    for _ in 0..extra {
                        bytes.push(self.next_byte()?);
                    }
                    out.push_str(&String::from_utf8_lossy(&bytes));
                }
            }
        }
    }
}

impl<R: BufRead> Iterator for Documents<R> {
    type Item = String;

    /// Scans forward for the next `"input_text"` key and returns its value.
    fn next(&mut self) -> Option<String> {
        const KEY: &[u8] = b"\"input_text\"";
        let mut matched = 0usize;
        loop {
            let b = self.next_byte()?;
            if b == KEY[matched] {
                matched += 1;
                if matched == KEY.len() {
                    // Skip whitespace and the colon, then the opening quote.
                    loop {
                        match self.next_byte()? {
                            b'"' => return self.string_body(),
                            b':' | b' ' | b'\n' | b'\r' | b'\t' => {}
                            _ => break,
                        }
                    }
                    matched = 0;
                }
            } else {
                matched = usize::from(b == KEY[0]);
            }
        }
    }
}

/// What a corpus turned into, and how much of it survived the vocabulary cut.
pub struct Stream {
    pub tokens: Vec<u32>,
    pub distinct_pieces: usize,
    /// Share of tokens that fell outside the kept vocabulary.
    pub unknown_share: f64,
    pub documents: usize,
    /// The most frequent pieces, spelled out. A vocabulary that looks wrong
    /// looks wrong here, before it costs a run.
    pub commonest: Vec<String>,
}

/// Reads `path`, segments it, and remaps onto a compact vocabulary of `vocab`
/// ids: the `vocab - 1` most frequent pieces keep an id of their own and
/// everything else collapses into the last one.
///
/// The model that ships with this corpus has 262144 pieces. An output head is
/// `vocab * d_model`, and the per-token expert is `vocab * vocab`, so the full
/// piece inventory is not a thing this architecture can carry yet; truncation is
/// the honest way to run at all, and `unknown_share` is what has to be reported
/// alongside any number produced this way.
///
/// The file is streamed and only the id vector is retained, so memory is set by
/// `limit` rather than by the size of the corpus.
pub fn stream(corpus: &Path, model: &Path, limit: usize, vocab: usize) -> std::io::Result<Stream> {
    assert!(vocab >= 2, "a corpus vocabulary needs at least one real id");
    let tk = Tokenizer::load(model)?;
    let mut raw: Vec<u32> = Vec::with_capacity(limit.min(1 << 22));
    let mut documents = 0usize;
    for text in Documents::open(corpus)? {
        documents += 1;
        tk.encode(&text, &mut raw);
        if raw.len() >= limit {
            raw.truncate(limit);
            break;
        }
    }
    let mut count: HashMap<u32, u64> = HashMap::new();
    for t in &raw {
        *count.entry(*t).or_insert(0) += 1;
    }
    let distinct_pieces = count.len();
    let mut ranked: Vec<(u32, u64)> = count.into_iter().collect();
    // Frequency first, then piece id, so the mapping does not depend on the
    // iteration order of a hash map.
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let keep = vocab - 1;
    let mut remap: HashMap<u32, u32> = HashMap::with_capacity(keep);
    for (slot, (piece, _)) in ranked.iter().take(keep).enumerate() {
        remap.insert(*piece, slot as u32);
    }
    let unknown = (vocab - 1) as u32;
    let mut misses = 0u64;
    let tokens: Vec<u32> = raw
        .iter()
        .map(|t| match remap.get(t) {
            Some(id) => *id,
            None => {
                misses += 1;
                unknown
            }
        })
        .collect();
    let unknown_share = if tokens.is_empty() {
        0.0
    } else {
        misses as f64 / tokens.len() as f64
    };
    let commonest = ranked
        .iter()
        .take(12)
        .map(|(piece, _)| tk.piece(*piece).to_string())
        .collect();
    Ok(Stream {
        tokens,
        distinct_pieces,
        unknown_share,
        documents,
        commonest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn model() -> Option<Tokenizer> {
        // The model lives beside the workspace, not in it; skip rather than fail
        // when a checkout does not have it.
        for candidate in ["tokenizer.model", "../tokenizer.model"] {
            let p = Path::new(candidate);
            if p.exists() {
                return Tokenizer::load(p).ok();
            }
        }
        None
    }

    /// Token ids from the reference `sentencepiece` implementation on this exact
    /// model file. If the encoder drifts, these fail — which is the only way to
    /// notice, since a wrong segmentation still produces perfectly valid ids and
    /// perfectly plausible losses.
    #[test]
    fn matches_the_reference_implementation() {
        let Some(tk) = model() else { return };
        let cases: &[(&str, &[u32])] = &[
            (
                "The Project Gutenberg Etext of The Declaration of Independence.",
                &[
                    727, 7306, 75530, 632, 1915, 500, 615, 38034, 500, 31429, 261835,
                ],
            ),
            ("hello world", &[41702, 1299]),
            (" leading space", &[4757, 2680]),
            ("double  space", &[4142, 465, 12796]),
            ("tab\there", &[14384, 35, 1733]),
            ("newline\nhere", &[125862, 218, 1733]),
            (
                "MiXeD CaSe 12345",
                &[
                    28879, 157291, 261859, 10384, 3346, 261813, 261846, 261853, 261869, 261873,
                    261874,
                ],
            ),
            (
                "naïve café — em-dash",
                &[1022, 403, 383, 522, 37818, 2067, 965, 261840, 63975],
            ),
            ("", &[]),
            ("a", &[261816]),
        ];
        for (text, want) in cases {
            let mut got = Vec::new();
            tk.encode(text, &mut got);
            assert_eq!(&got[..], *want, "segmentation drifted on {text:?}");
        }
    }

    #[test]
    fn the_documents_reader_survives_chunk_boundaries() {
        // The key, the escapes and the multi-byte characters all have to work
        // when a read lands in the middle of them, which is the failure a
        // whole-file loader never sees and a streaming one hits constantly.
        let json = r#"[{"input_text": "first\nline"},{"input_text": "café 😀 end"}]"#;
        for chunk in [1usize, 2, 3, 7, 64] {
            let reader = ChunkedReader {
                data: json.as_bytes(),
                at: 0,
                chunk,
            };
            let docs: Vec<String> = Documents {
                reader: BufReader::new(reader),
                buf: Vec::new(),
                at: 0,
            }
            .collect();
            assert_eq!(
                docs,
                vec!["first\nline", "café 😀 end"],
                "chunk size {chunk}"
            );
        }
    }

    /// Hands out at most `chunk` bytes per read, to force boundaries anywhere.
    struct ChunkedReader<'a> {
        data: &'a [u8],
        at: usize,
        chunk: usize,
    }

    impl Read for ChunkedReader<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let n = self.chunk.min(out.len()).min(self.data.len() - self.at);
            out[..n].copy_from_slice(&self.data[self.at..self.at + n]);
            self.at += n;
            Ok(n)
        }
    }
}

#[cfg(test)]
mod bulk {
    use super::*;
    /// Dumps the encoding of the first documents of the corpus so it can be
    /// diffed against the reference implementation in bulk. Ignored by default:
    /// it needs files that live outside the workspace.
    #[test]
    #[ignore]
    fn dump_for_reference_diff() {
        let (model, corpus) = (
            Path::new("../tokenizer.model"),
            Path::new("../stage_1_1-32001.json"),
        );
        if !model.exists() {
            return;
        }
        let tk = Tokenizer::load(model).unwrap();
        let mut out = String::new();
        for text in Documents::open(corpus).unwrap().take(4000) {
            let mut ids = Vec::new();
            tk.encode(&text, &mut ids);
            out.push_str(
                &ids.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
            out.push('\n');
        }
        std::fs::write("/tmp/rust_tokens.txt", out).unwrap();
    }
}
