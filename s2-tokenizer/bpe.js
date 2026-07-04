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
// Returns { merges: [[a, b, newId], ...], vocab: Uint8Array[] (index = token id) }.
async function trainBPE(byteArrays, vocabSize, onProgress) {
    const ids = [];
    byteArrays.forEach((arr, i) => {
        if (i > 0) ids.push(0);
        for (const b of arr) ids.push(b);
    });

    const n = ids.length;
    const next = new Int32Array(n).fill(-1);
    const prev = new Int32Array(n).fill(-1);
    const val = new Int32Array(n);
    const alive = new Uint8Array(n).fill(1);
    for (let i = 0; i < n; i++) {
        val[i] = ids[i];
        next[i] = i + 1 < n ? i + 1 : -1;
        prev[i] = i > 0 ? i - 1 : -1;
    }

    const counts = new Map();
    const positions = new Map();

    function addPair(p) {
        if (p === -1 || next[p] === -1) return;
        const key = pairKey(val[p], val[next[p]]);
        counts.set(key, (counts.get(key) || 0) + 1);
        let set = positions.get(key);
        if (!set) { set = new Set(); positions.set(key, set); }
        set.add(p);
    }

    function removePairAt(p) {
        if (p === -1 || next[p] === -1) return;
        const key = pairKey(val[p], val[next[p]]);
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

    for (let m = 0; m < maxMerges; m++) {
        let bestKey = -1, bestCount = 0;
        for (const [key, c] of counts) {
            if (c > bestCount) { bestCount = c; bestKey = key; }
        }
        if (bestKey === -1 || bestCount < 2) break;

        const [a, b] = unpackKey(bestKey);
        const newId = nextId++;
        vocab.push(concatBytes(vocab[a], vocab[b]));
        merges.push([a, b, newId]);

        const occurrences = Array.from(positions.get(bestKey) || []);
        positions.delete(bestKey);
        counts.delete(bestKey);

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

            addPair(prev[p]);
            addPair(p);
        }

        if (onProgress && (m % 25 === 0 || m === maxMerges - 1)) {
            onProgress(m + 1, maxMerges);
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }

    return { merges, vocab };
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
