// Vanilla byte-level BPE (no regex pretokenization — merges may span word/space
// boundaries), following the approach in Karpathy's minbpe "Basic" tokenizer.

function textToBytes(text) {
    return Array.from(new TextEncoder().encode(text));
}

function concatBytes(a, b) {
    const out = new Uint8Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
}

const PAIR_MULT = 100000;
function pairKey(a, b) { return a * PAIR_MULT + b; }
function unpackKey(key) {
    const a = Math.floor(key / PAIR_MULT);
    return [a, key - a * PAIR_MULT];
}

// Trains a joint BPE vocab over several byte-array corpora (concatenated with a
// 0x00 boundary byte, which never appears in real text and never wins a merge).
// `wordCounts` (default: each corpus's byte length, a rough proxy) is the actual
// word count for each language — the denominator of the fertility metric X =
// tokens/words (the standard NLP definition — always >= 1, since a word can't be
// represented by less than one token). At every step, the next merge is spent on
// whichever active language currently has the *highest* live fertility (live
// token count / words so far), so the budget continuously chases down the worst
// performer instead of being split by a fixed schedule. A language drops out
// once it runs out of repeated pairs, freeing the rest of the budget for the others.
// Also returns `mergesPerLang` (how many merges each language actually won) and
// `history` (fertility = liveTokens/wordCounts for every language, sampled through
// training) so the caller can show the adaptive allocation at work.
// `onProgress(done, total, history)` fires periodically mid-training with the
// same `history` array the return value carries — a caller can redraw a live
// chart from it before training finishes, since it's the same growing array.
// Returns { merges, vocab, mergesPerLang, history }.
async function trainBPE(byteArrays, vocabSize, onProgress, wordCounts) {
    const nLangs = byteArrays.length;
    if (!wordCounts) wordCounts = byteArrays.map(a => a.length);

    const ids = [];
    const langOf = [];
    byteArrays.forEach((arr, i) => {
        if (i > 0) { ids.push(0); langOf.push(-1); }
        for (const b of arr) { ids.push(b); langOf.push(i); }
    });

    const n = ids.length;
    const next = new Int32Array(n).fill(-1);
    const prev = new Int32Array(n).fill(-1);
    const val = new Int32Array(n);
    const alive = new Uint8Array(n).fill(1);
    const lang = Int8Array.from(langOf);
    for (let i = 0; i < n; i++) {
        val[i] = ids[i];
        next[i] = i + 1 < n ? i + 1 : -1;
        prev[i] = i > 0 ? i - 1 : -1;
    }

    const countsPerLang = Array.from({ length: nLangs }, () => new Map()); // countsPerLang[l]: key -> count
    const positions = new Map(); // key -> Set of positions (all langs)

    function addPair(p) {
        if (p === -1 || next[p] === -1 || lang[p] < 0) return; // lang[p] < 0: p is a boundary byte, never a real merge candidate
        const key = pairKey(val[p], val[next[p]]);
        const counts = countsPerLang[lang[p]];
        counts.set(key, (counts.get(key) || 0) + 1);
        let set = positions.get(key);
        if (!set) { set = new Set(); positions.set(key, set); }
        set.add(p);
    }

    function removePairAt(p) {
        if (p === -1 || next[p] === -1 || lang[p] < 0) return;
        const key = pairKey(val[p], val[next[p]]);
        const counts = countsPerLang[lang[p]];
        const c = counts.get(key);
        if (c !== undefined) {
            if (c <= 1) counts.delete(key); else counts.set(key, c - 1);
        }
        const set = positions.get(key);
        if (set) { set.delete(p); if (set.size === 0) positions.delete(key); }
    }

    for (let i = 0; i < n; i++) addPair(i);

    const merges = [];
    const vocab = [];
    for (let i = 0; i < 256; i++) vocab.push(new Uint8Array([i]));

    let nextId = 256;
    const maxMerges = vocabSize - 256;

    // Live token count per language, updated as merges collapse positions —
    // this is what "live token count / words" (live fertility) tracks during training.
    const liveTokens = byteArrays.map(a => a.length);
    const active = new Array(nLangs).fill(true);
    const mergesPerLang = new Array(nLangs).fill(0);
    const history = [];
    const HISTORY_STRIDE = 25; // matches the onProgress cadence below, so callers can render live
    let done = 0;
    const snapshot = () => ({ merge: done, ratios: wordCounts.map((w, l) => liveTokens[l] / w) });
    history.push(snapshot());

    for (let m = 0; m < maxMerges; m++) {
        // Spend this merge on whichever active language is currently worst off (highest fertility).
        let chosenLang = -1, worstFertility = -Infinity;
        for (let l = 0; l < nLangs; l++) {
            if (!active[l]) continue;
            const fertility = liveTokens[l] / wordCounts[l];
            if (fertility > worstFertility) { worstFertility = fertility; chosenLang = l; }
        }
        if (chosenLang === -1) break; // no language has any repeated pair left

        let bestKey = -1, bestCount = 0;
        for (const [key, c] of countsPerLang[chosenLang]) {
            if (c > bestCount) { bestCount = c; bestKey = key; }
        }
        if (bestKey === -1 || bestCount < 2) {
            active[chosenLang] = false;
            continue;
        }

        const [a, b] = unpackKey(bestKey);
        const newId = nextId++;
        vocab.push(concatBytes(vocab[a], vocab[b]));
        merges.push([a, b, newId]);
        mergesPerLang[chosenLang]++;

        const occurrences = Array.from(positions.get(bestKey) || []);
        positions.delete(bestKey);
        for (const counts of countsPerLang) counts.delete(bestKey);

        for (const p of occurrences) {
            if (!alive[p] || next[p] === -1) continue;
            const q = next[p];
            if (!alive[q] || val[p] !== a || val[q] !== b) continue;

            removePairAt(prev[p]);
            removePairAt(q);

            const r = next[q];
            next[p] = r;
            if (r !== -1) prev[r] = p;
            val[p] = newId;
            alive[q] = 0;
            liveTokens[lang[p]]--; // one fewer live token for whichever language this occurrence belongs to

            addPair(prev[p]);
            addPair(p);
        }

        done++;
        if (done % HISTORY_STRIDE === 0 || done === maxMerges) history.push(snapshot());
        if (onProgress && (done % 25 === 0 || done === maxMerges)) {
            onProgress(done, maxMerges, history);
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }
    history.push(snapshot()); // failsafe: capture the final state even if the loop broke early (all languages exhausted)

    return { merges, vocab, mergesPerLang, history };
}

// Encodes raw bytes using previously learned merges (applied in learned order).
function encodeBytes(byteArray, merges) {
    const n = byteArray.length;
    if (n === 0) return [];

    const rank = new Map();
    merges.forEach(([a, b], i) => { rank.set(pairKey(a, b), i); });

    const next = new Int32Array(n).fill(-1);
    const prev = new Int32Array(n).fill(-1);
    const val = new Int32Array(n);
    const alive = new Uint8Array(n).fill(1);
    for (let i = 0; i < n; i++) {
        val[i] = byteArray[i];
        next[i] = i + 1 < n ? i + 1 : -1;
        prev[i] = i > 0 ? i - 1 : -1;
    }

    const positions = new Map();

    function tryAddPair(p) {
        if (p === -1 || next[p] === -1) return;
        const key = pairKey(val[p], val[next[p]]);
        if (!rank.has(key)) return;
        let set = positions.get(key);
        if (!set) { set = new Set(); positions.set(key, set); }
        set.add(p);
    }

    function tryRemovePair(p) {
        if (p === -1 || next[p] === -1) return;
        const key = pairKey(val[p], val[next[p]]);
        const set = positions.get(key);
        if (set) { set.delete(p); if (set.size === 0) positions.delete(key); }
    }

    for (let i = 0; i < n; i++) tryAddPair(i);

    while (positions.size > 0) {
        let bestKey = -1, bestRank = Infinity;
        for (const key of positions.keys()) {
            const r = rank.get(key);
            if (r < bestRank) { bestRank = r; bestKey = key; }
        }
        if (bestKey === -1) break;

        const [a, b] = unpackKey(bestKey);
        const newId = 256 + bestRank;

        const occurrences = Array.from(positions.get(bestKey) || []);
        positions.delete(bestKey);

        for (const p of occurrences) {
            if (!alive[p] || next[p] === -1) continue;
            const q = next[p];
            if (!alive[q] || val[p] !== a || val[q] !== b) continue;

            tryRemovePair(prev[p]);
            tryRemovePair(q);

            const r = next[q];
            next[p] = r;
            if (r !== -1) prev[r] = p;
            val[p] = newId;
            alive[q] = 0;

            tryAddPair(prev[p]);
            tryAddPair(p);
        }
    }

    const out = [];
    for (let i = 0; i < n; i++) if (alive[i]) out.push(val[i]);
    return out;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { textToBytes, concatBytes, trainBPE, encodeBytes };
}
