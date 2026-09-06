/* Session 8 attention timeline -- narrative content and cost models.
 *
 * Dates here are duplicated from sources.json, which is the authority and is machine-checked
 * against arXiv by verify_sources.py. If the two disagree, sources.json wins and the check
 * fails loudly. Nothing on this page is dated from memory.
 *
 * bill[]  which cost the mechanism attacks -- this is what colours the timeline, and watching
 *         the colours migrate down the years is the entire point of ordering by date.
 *           compute  the T^2 score matrix
 *           kv       the per-user KV cache
 *           pos      position representation and length
 *           sys      serving/systems layer, no change to the mathematics
 * kv      cache model, evaluated against the notes' section-10 yardstick
 * comp    compute model
 */

const BASE = { layers: 48, qHeads: 64, headDim: 128, bytes: 2, kvHeadsGQA: 8 };

const MECHANISMS = [
  {
    id: 'learned-absolute', date: '2017-05-08', name: 'Learned absolute positions',
    src: 'Convolutional Sequence to Sequence Learning', arxiv: '1705.03122',
    bill: ['pos'], kv: { t: 'none' }, comp: 'none', family: 'position',
    problem: 'A model that processes every token in parallel has no idea which token came first. Order has to be injected from outside.',
    mech: 'Add one learned vector per absolute slot -- row 0 for position 0, row 1 for position 1 -- and train it like any other embedding table.',
    buys: 'Trivial to implement, and the model discovers whatever positional structure the task actually needs rather than being told.',
    costs: 'A table has a hard edge. Row 2049 of a 2048-row table does not exist -- not degraded, absent. And every row trains independently, so nothing learned at position 100 transfers to position 900.',
    pick: 'Fixed, short, known context where the simplest thing wins -- a 512-token classification encoder. Essentially never for a modern decoder.',
    broke: 'The hard length wall. Every position mechanism after this exists to remove it.',
    note: 'Five weeks older than the Transformer. Commonly misattributed to it because that is where most people first meet the idea.'
  },
  {
    id: 'standard-attention', date: '2017-06-12', name: 'Scaled dot-product attention',
    src: 'Attention Is All You Need', arxiv: '1706.03762',
    bill: [], kv: { t: 'mha' }, comp: 'quad', family: 'foundation', anchor: true,
    problem: 'Recurrence forces sequential computation and squeezes long-range signal through one hidden state.',
    mech: 'Every token emits a query, a key and a value. Score each query against every key, scale by sqrt(d_k), mask the future, softmax, and take the weighted sum of values. All positions at once, several heads in parallel.',
    buys: 'Exact all-to-all access. Any token can read any earlier token directly, and each query forms its own fresh distribution over the whole past. Training parallelises across the sequence.',
    costs: 'T^2 comparisons, and a KV cache that grows linearly with context for every concurrent user. It was never wrong -- it was expensive.',
    pick: 'Whenever you can afford it. It is still the highest-quality attention we have. This whole timeline is about affording it, not replacing it.',
    broke: 'Sent two bills that come due at different moments: compute at T^2 during training and prefill, cache at T during generation.'
  },
  {
    id: 'sinusoidal', date: '2017-06-12', name: 'Sinusoidal positions',
    src: 'Attention Is All You Need', arxiv: '1706.03762',
    bill: ['pos'], kv: { t: 'none' }, comp: 'none', family: 'position',
    problem: 'A learned table cannot be evaluated past its last row.',
    mech: 'Replace the table with a function: sines and cosines of position at geometrically spaced frequencies, added to the input embedding.',
    buys: 'Defined at every integer position, so there is no table to fall off. Costs zero parameters. The first real instance of computing position instead of storing it.',
    costs: 'Defined is not the same as trained. The model still only ever saw patterns from inside its training range, and quality falls off outside it regardless of the formula. Because it is added at the input, positional information has to survive every layer intact.',
    pick: 'Rarely on its own today -- RoPE does the same job better at every layer. Historically the important half of this paper for long context.',
    broke: 'Established the principle without delivering the practice: extrapolation stayed poor in practice.'
  },
  {
    id: 'transformer-xl', date: '2019-01-09', name: 'Transformer-XL segment recurrence',
    src: 'Transformer-XL', arxiv: '1901.02860',
    bill: ['pos', 'kv'], kv: { t: 'window', w: 8192 }, comp: 'window', family: 'state',
    problem: 'Chop a long document into chunks and everything at each boundary is simply lost.',
    mech: 'Cache the previous segment\'s hidden states and let the current segment attend into them, with relative position encoding so the reused states stay meaningful.',
    buys: 'Context reaching past one window without paying T^2 over the whole document. Introduced relative position encoding to mainstream LMs.',
    costs: 'Memory grows with how many segments you retain, and gradients are stopped at the boundary -- so the model can use long dependencies but cannot really learn them end-to-end.',
    pick: 'Streaming over long documents where a hard boundary would be worse than a compressed one.',
    broke: 'Nothing improved the quality of what crosses the boundary. That question is still open -- it is exactly what section 14\'s Memory Stream re-opens.'
  },
  {
    id: 'sparse-transformer', date: '2019-04-23', name: 'Sparse Transformer',
    src: 'Generating Long Sequences with Sparse Transformers', arxiv: '1904.10509',
    bill: ['compute'], kv: { t: 'none' }, comp: 'sqrt', family: 'sparse',
    problem: 'T^2 is unaffordable once sequences reach image or audio scale.',
    mech: 'Give each head a fixed pattern -- a local band plus a strided skip -- so it attends to a subset of positions rather than all of them.',
    buys: 'Compute drops from T^2 to roughly T*sqrt(T). Long sequences become trainable at all.',
    costs: 'The pattern is fixed and content-blind. If the token you needed is not on your stride, you cannot see it -- not this time, not ever, regardless of how relevant it was.',
    pick: 'Data with genuine positional locality (images, audio) where a fixed pattern happens to match the structure.',
    broke: 'Content-blindness. Every later sparse method is about choosing keys by content instead of by position.'
  },
  {
    id: 't5-relative-bias', date: '2019-10-23', name: 'T5 relative position bias',
    src: 'Exploring the Limits of Transfer Learning (T5)', arxiv: '1910.10683',
    bill: ['pos'], kv: { t: 'none' }, comp: 'none', family: 'position',
    problem: 'Absolute position says where a token is. Attention wants to know how far apart two tokens are.',
    mech: 'Learn one scalar per relative-distance bucket and add it to the score before softmax. Nearby distances get their own bucket; far ones share.',
    buys: 'Relative by construction, so shifting the whole sequence changes nothing. Bucketing shares statistics across distances, which generalises far better than one independent row per position.',
    costs: 'Still a table, just of buckets. The last bucket saturates, so past it every distance looks identical. And it is a per-head learned parameter set, not free.',
    pick: 'Encoder-decoder models at moderate context.',
    broke: 'Proved that a bias added to the score works -- which ALiBi then strips down to a fixed slope with no parameters at all.'
  },
  {
    id: 'mqa', date: '2019-11-06', name: 'Multi-Query Attention',
    src: 'Fast Transformer Decoding: One Write-Head is All You Need', arxiv: '1911.02150',
    bill: ['kv'], kv: { t: 'heads', h: 1 }, comp: 'quad', family: 'kv',
    problem: 'During generation the KV cache is a private per-user cost, and at any real batch size it, not the weights, is what fills the accelerator.',
    mech: 'Keep all the query heads, but have every one of them read a single shared key/value head.',
    buys: 'Cache shrinks by the full head count -- commonly 32x to 64x. Decoding stops being memory-bandwidth-bound.',
    costs: 'A real quality drop. Every head now has to search the same K/V space, so heads lose the independent retrieval behaviour that multi-head attention exists to provide. Training instability was reported too.',
    pick: 'Decode-bound serving where cache is unambiguously the binding constraint and the quality cost is acceptable.',
    broke: 'Overshot. The quality loss is precisely what motivated GQA three and a half years later.',
    note: 'Frequently mis-dated into the 2023 GQA era, because that is when it became widely deployed rather than when it was proposed.'
  },
  {
    id: 'longformer', date: '2020-04-10', name: 'Sliding window + global tokens',
    src: 'Longformer: The Long-Document Transformer', arxiv: '2004.05150',
    bill: ['compute', 'kv'], kv: { t: 'window', w: 4096 }, comp: 'window', family: 'sparse',
    problem: 'Fixed strided sparsity is awkward for text, which wants strong locality plus a few global anchors.',
    mech: 'Each token attends to a window of w neighbours; a handful of designated tokens attend to everything and are attended by everything.',
    buys: 'Compute and cache both become linear in T. Conceptually simple and easy to implement.',
    costs: 'Information moves only w positions per layer, so anything cross-document needs depth to propagate. And which tokens get to be global is a human design decision the model has no say in.',
    pick: 'Long documents with local structure and known anchors -- a classification token, a question prefix.',
    broke: 'Nothing carries genuinely long-range detail, and a naive sliding window turns out to collapse in streaming. That failure gets diagnosed in 2023.'
  },
  {
    id: 'linear-attention', date: '2020-06-29', name: 'Linear attention',
    src: 'Transformers are RNNs', arxiv: '2006.16236',
    bill: ['compute', 'kv'], kv: { t: 'state' }, comp: 'lin', family: 'state',
    problem: 'Softmax\'s shared denominator ties every score to every other, which forces you to keep each old key around until the query arrives.',
    mech: 'Replace exp(q.k) with a factorable feature map phi(q).phi(k). The sum then regroups into S = sum of phi(k)v^T -- one fixed-size state -- and the output is a read from S.',
    buys: 'The cache stops growing entirely: one state matrix after ten tokens, after a million tokens. Sequence work becomes linear and decoding becomes constant-time.',
    costs: 'The state is a lossy compression of everything. Different memories interfere, recall of one specific old token degrades badly, and the add-only write cannot revise an association it already holds.',
    pick: 'Very long streams where exact recall of arbitrary earlier tokens is not the job -- or, far more commonly, as the cheap majority of layers in a hybrid.',
    broke: 'Memory quality. Sections 5 and 6 of the notes are the direct consequence.'
  },
  {
    id: 'delta-rule', date: '2021-02-22', name: 'The delta rule as a state write',
    src: 'Linear Transformers Are Secretly Fast Weight Programmers', arxiv: '2102.11174',
    bill: ['kv'], kv: { t: 'state' }, comp: 'lin', family: 'state',
    problem: 'An add-only state carries its old contribution forward. If key A currently returns 40 and should now return 55, adding the whole new answer gives 95.',
    mech: 'Read what the state currently returns for this key, subtract it from what it should return, and write only that difference.',
    buys: 'Turns a pile-up into an updateable memory. The state can now correct an association rather than only accumulate onto it.',
    costs: 'The write is inherently sequential -- each step depends on the state the previous step produced -- which made it untrainable at scale for three more years.',
    pick: 'Any fixed-state layer you intend to train seriously, but only once you have the parallel algorithm.',
    broke: 'Parallelisation. Not solved until 2024.',
    note: 'The delta rule itself is Widrow-Hoff, around 1960. This date is its arrival as a linear-attention state write, which is the claim this timeline is making.'
  },
  {
    id: 'rope', date: '2021-04-20', name: 'RoPE (rotary positions)',
    src: 'RoFormer', arxiv: '2104.09864',
    bill: ['pos'], kv: { t: 'none' }, comp: 'none', family: 'position', anchor: true,
    problem: 'Sinusoidal position is added at the input and dilutes through depth; learned tables have a wall.',
    mech: 'Treat pairs of dimensions as 2D arrows and rotate the query and key by an angle proportional to their positions. Because a dot product depends on the angle between arrows, both absolute rotations cancel and only i-j survives.',
    buys: 'Relative distance by construction, re-applied inside every attention layer rather than once at the input, defined at any position, and costing zero parameters.',
    costs: 'Defined is still not trained. Quality falls off past the training length, and the frequency band is implicitly calibrated to the range the model actually saw.',
    pick: 'The default for decoder LLMs. There is essentially no reason to choose a learned absolute table over it today.',
    broke: 'Created an entire industry. Position Interpolation, NTK-aware scaling and YaRN all exist to make RoPE work past the length it was trained at.'
  },
  {
    id: 'alibi', date: '2021-08-27', name: 'ALiBi',
    src: 'Train Short, Test Long', arxiv: '2108.12409',
    bill: ['pos'], kv: { t: 'none' }, comp: 'none', family: 'position',
    problem: 'Even RoPE degrades past its training length, and everyone wanted train-short-test-long to actually work.',
    mech: 'Remove position embeddings entirely. Subtract a fixed per-head slope times the token distance from each score before softmax.',
    buys: 'It genuinely extrapolates -- train at 1K, run considerably longer with stable perplexity. Zero parameters, a few lines of code, no cache impact.',
    costs: 'It bakes in a monotonic recency prior. The model is structurally penalised for attending far away, so tasks that need exact retrieval from deep in the context suffer. Every head decays; only the rate differs.',
    pick: 'Streaming or long-document language modelling where recency genuinely dominates. A poor choice for long-context retrieval.',
    broke: 'Traded retrieval quality for extrapolation. The field largely chose RoPE-plus-extension instead, which is why ALiBi is less common now than its 2021 reception suggested.'
  },
  {
    id: 'flashattention', date: '2022-05-27', name: 'FlashAttention',
    src: 'FlashAttention: Fast and Memory-Efficient Exact Attention', arxiv: '2205.14135',
    bill: ['compute', 'sys'], kv: { t: 'none' }, comp: 'quad', family: 'systems', anchor: true,
    problem: 'Everyone assumed attention was FLOP-bound. It was actually bandwidth-bound -- the real cost was writing the T x T score matrix out to HBM and reading it back.',
    mech: 'Tile Q, K and V into on-chip SRAM, compute softmax incrementally over tiles, and never materialise the full score matrix in HBM at all. Recompute it during the backward pass instead of storing it.',
    buys: 'Two to four times the wall-clock speed and memory dropping from O(T^2) to O(T) -- with bit-identical outputs. Not an approximation of any kind.',
    costs: 'Mathematically none, which is the point. The cost is engineering: hand-written kernels retuned for each GPU generation. And it does not reduce FLOPs, so at very long T the quadratic compute still eventually binds.',
    pick: 'Always. There is no configuration where you would prefer the naive kernel.',
    broke: 'Nothing -- and that is why it matters here. It made exact attention so much cheaper that most of the 2020 approximation wave (Linformer, Performer, Reformer) stopped being worth the quality risk. It is the timeline\'s proof that not every saving requires giving something up.'
  },
  {
    id: 'gqa', date: '2023-05-22', name: 'Grouped-Query Attention',
    src: 'GQA', arxiv: '2305.13245',
    bill: ['kv'], kv: { t: 'heads', h: 8 }, comp: 'quad', family: 'kv', anchor: true,
    problem: 'MQA saves the most cache but costs quality; MHA keeps quality but pays the full cache.',
    mech: 'Put the query heads into G groups and give each group one shared K/V head. G equal to the head count is MHA; G of 1 is MQA; everything useful is in between.',
    buys: 'A dial rather than a choice. Around 8 groups typically holds near-MHA quality at roughly an eighth of the cache -- and it can be uptrained from an existing MHA checkpoint for about 5% of pretraining compute rather than trained fresh.',
    costs: 'It lowers the slope of the line without flattening it. The cache is still linear in T, so at a million tokens it still holds a million positions. Constant-factor relief only.',
    pick: 'The default, and almost every modern decoder uses it. But treat it as a baseline you build on, not a long-context answer.',
    broke: 'Nothing, and that is exactly the problem -- it is insufficient alone, which is what forces everything after it.'
  },
  {
    id: 'position-interpolation', date: '2023-06-27', name: 'Position Interpolation',
    src: 'Extending Context Window via Positional Interpolation', arxiv: '2306.15595',
    bill: ['pos'], kv: { t: 'none' }, comp: 'none', family: 'position',
    problem: 'Asking RoPE for a rotation past the training length produces angle patterns the model has never learned to read.',
    mech: 'Do not extrapolate -- interpolate. Divide the position indices by a scale factor so that 8K positions land inside the 2K range the model was actually trained on.',
    buys: 'Works with a surprisingly small amount of fine-tuning, around a thousand steps, and is a few lines of change.',
    costs: 'Squeezing every frequency by the same factor crushes the high-frequency channels that encode local distance, so adjacent tokens become harder to tell apart -- you buy reach by blurring detail. It also still requires fine-tuning.',
    pick: 'Superseded. Use YaRN. It is here because the next two entries do not make sense without it.',
    broke: 'The high-frequency loss is the direct and stated cause of NTK-aware scaling.'
  },
  {
    id: 'ntk-aware', date: '2023-06', datePrec: 'month', name: 'NTK-aware scaled RoPE',
    src: 'Community post by u/bloc97 on r/LocalLLaMA', arxiv: null,
    url: 'https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/',
    bill: ['pos'], kv: { t: 'none' }, comp: 'none', family: 'position',
    problem: 'Position Interpolation squeezes all frequencies equally and destroys the local resolution that high-frequency channels carry.',
    mech: 'Scale the RoPE base rather than the position index, so low-frequency channels stretch a great deal and high-frequency channels barely move at all.',
    buys: 'Extends context with no fine-tuning whatsoever, while keeping local detail that PI blurs. You can apply it to a checkpoint you did not train.',
    costs: 'Heuristic and hand-tuned, with no formal analysis behind the scaling choice, and quality still degrades at large extension factors.',
    pick: 'Quick extension of an existing checkpoint when fine-tuning is not an option.',
    broke: 'Worked well enough to demand a principled version, which is YaRN.',
    note: 'Not a paper. A forum post, never submitted to arXiv, with no DOI and no verifiable timestamp -- so it is dated to the month only. Its position after Position Interpolation is established by YaRN\'s own text describing it as a fix for PI\'s high-frequency loss, not by a calendar. An agent asked to cite this will happily invent an arXiv ID; there is not one.'
  },
  {
    id: 'yarn', date: '2023-08-31', name: 'YaRN',
    src: 'YaRN: Efficient Context Window Extension', arxiv: '2309.00071',
    bill: ['pos'], kv: { t: 'none' }, comp: 'none', family: 'position',
    problem: 'NTK-aware scaling works but is a heuristic, and quality still slips at large factors.',
    mech: 'Treat frequency bands differently depending on whether their wavelength fits inside the trained window (NTK-by-parts), and add a temperature correction to the attention logits to compensate for the entropy shift that stretching causes.',
    buys: 'The best quality per fine-tuning token in the RoPE-extension family -- roughly ten times less data and two and a half times fewer steps than PI.',
    costs: 'Still fundamentally stretching a model trained short, so the ceiling remains. Needs some fine-tuning, and adds hyperparameters that have to be set per model.',
    pick: 'Extending an existing RoPE checkpoint. The most widely used member of this family.',
    broke: 'Nothing internally -- but it is Road 1, and Road 1 has a ceiling nobody has removed.',
    note: 'The arXiv ID starts 2309, so this is very widely cited as September 2023. The v1 submission is 31 August 2023. The clearest example on this timeline of the ID prefix not being the date.'
  },
  {
    id: 'paged-attention', date: '2023-09-12', name: 'PagedAttention',
    src: 'Efficient Memory Management for LLM Serving with PagedAttention', arxiv: '2309.06180',
    bill: ['sys'], kv: { t: 'none' }, comp: 'none', family: 'systems',
    problem: 'The KV cache was allocated as one contiguous block per sequence, sized for the worst case. Most of it sat empty -- reported waste of 60 to 80 percent.',
    mech: 'Borrow virtual memory. Store the cache in fixed-size blocks that need not be contiguous, keep a block table per sequence, and share blocks between sequences with copy-on-write.',
    buys: 'Two to four times the throughput at identical memory, purely from packing. Shared prefixes across requests become nearly free.',
    costs: 'It does not make the cache smaller, only better packed -- so it does nothing for a single long sequence, which is the case long-context work actually cares about.',
    pick: 'Any multi-user serving deployment. It is orthogonal to every architectural choice on this timeline and composes with all of them.',
    broke: 'Nothing. It is here as the reminder that not every answer to a bill is an architectural one.'
  },
  {
    id: 'attention-sinks', date: '2023-09-29', name: 'Attention sinks',
    src: 'Efficient Streaming LMs with Attention Sinks', arxiv: '2309.17453',
    bill: ['kv'], kv: { t: 'window', w: 4096 }, comp: 'window', family: 'sparse',
    problem: 'A sliding-window model streaming past its window does not degrade gracefully -- it collapses, and it collapses at the exact moment the very first tokens fall out of the cache.',
    mech: 'Pin the first few tokens in the cache permanently and slide the window over everything after them.',
    buys: 'Stable generation over millions of tokens with a fixed-size cache, for the cost of about four extra entries. Nearly free.',
    costs: 'It stabilises without extending. The model still only sees window-many recent tokens plus four ancient ones, so the middle of the stream remains unreachable. It is not a long-context method.',
    pick: 'Unbounded streaming generation under a hard memory ceiling -- a chat session that must never restart.',
    broke: 'Nothing. Its value was diagnostic: softmax weights must sum to one, so when no key is a good match the model needs somewhere to dump the leftover mass, and it learns to dump it on the first tokens. Evict them and that mass is forced onto tokens that actually matter, corrupting the distribution.'
  },
  {
    id: 'mistral-swa', date: '2023-10-10', name: 'Sliding window ships in a frontier LLM',
    src: 'Mistral 7B', arxiv: '2310.06825',
    bill: ['kv', 'compute'], kv: { t: 'window', w: 4096 }, comp: 'window', family: 'sparse',
    problem: 'Sliding-window attention had lived in encoders and long-document models for three years without becoming a decoder default.',
    mech: 'A 4096-token window in every layer, with the cache capped at the window size. Stacking layers gives an effective receptive field of roughly window times depth.',
    buys: 'A cache that stops growing at a known constant, in a model that was competitive with much larger ones. It is what made bounded-cache decoders mainstream.',
    costs: 'The same trade sliding windows always make -- distant exact retrieval degrades, and the receptive-field argument is about information propagating through depth, not about any single layer seeing far.',
    pick: 'Bounded-memory serving where most dependencies are reasonably local.',
    broke: 'Nothing new. Included as a deployment milestone rather than an invention -- the mechanism is Sparse Transformer and Longformer, three and four years earlier.'
  },
  {
    id: 'mamba', date: '2023-12-01', name: 'Mamba (selective state space)',
    src: 'Mamba: Linear-Time Sequence Modeling', arxiv: '2312.00752',
    bill: ['compute', 'kv'], kv: { t: 'state' }, comp: 'lin', family: 'state',
    problem: 'Linear attention writes every token into its state with the same rule, so the state cannot decide what deserves to be remembered.',
    mech: 'A state space model whose transition and input matrices are functions of the current token, so the state can selectively retain or forget -- plus a hardware-aware parallel scan that makes it fast in practice.',
    buys: 'Fixed state and linear time, with selectivity closing much of the quality gap to attention on a wide range of tasks. Arrived at the same destination as linear attention from an entirely different direction.',
    costs: 'Still a fixed-size summary, and it loses specifically on the tasks attention is best at -- copying, induction, exact retrieval of an arbitrary earlier token. Being selective is not the same as being able to look things up.',
    pick: 'As the cheap majority of layers in a hybrid, not on its own.',
    broke: 'The recall gap, which is what makes hybrid depth schedules necessary rather than optional.'
  },
  {
    id: 'mla', date: '2024-05-07', name: 'Multi-head Latent Attention',
    src: 'DeepSeek-V2', arxiv: '2405.04434',
    bill: ['kv'], kv: { t: 'latent', dim: 576 }, comp: 'quad', family: 'kv',
    problem: 'GQA buys cache savings by deleting K/V heads, and head diversity is the thing multi-head attention exists to provide.',
    mech: 'Project keys and values down into a shared low-rank latent vector, cache only that latent, and reconstruct per-head keys and values at use time. RoPE is carried in a small separate cached component because rotation does not commute with the up-projection.',
    buys: 'A cache smaller than GQA\'s while reporting quality better than full MHA -- because you compress the representation instead of removing heads. All heads stay distinct.',
    costs: 'Substantially more complex. The up-projection is real extra compute at every decode step, and the decoupled-RoPE split is an awkward structural wart that exists purely because the maths forces it.',
    pick: 'Training from scratch, when cache is the binding constraint and you can carry the implementation complexity.',
    broke: 'Still linear in T. A smaller constant on a line that keeps climbing.',
    note: 'The decoupled RoPE component is what section 8 of the notes is pointing at with "RoPE applied to the last 64 dimensions".'
  },
  {
    id: 'ssd', date: '2024-05-31', name: 'State Space Duality',
    src: 'Transformers are SSMs (Mamba-2)', arxiv: '2405.21060',
    bill: [], kv: { t: 'state' }, comp: 'lin', family: 'state',
    problem: 'Linear attention and state space models were two literatures solving the same problem with separate vocabularies and separate tricks.',
    mech: 'Show both are instances of one structured semiseparable matrix transform, and derive a chunkwise algorithm from that view.',
    buys: 'Two to eight times faster than Mamba\'s scan, and more importantly it lets optimisations built for attention -- tensor cores, matmul kernels -- apply to state models.',
    costs: 'Primarily a theoretical unification. The accompanying model is an increment rather than a leap.',
    pick: 'Not a choice you make. It is the reason later work mixes the two families without ceremony.',
    broke: 'Nothing. It removed a wall between two research communities, which is why every hybrid after this borrows freely from both.'
  },
  {
    id: 'parallel-deltanet', date: '2024-06-10', name: 'DeltaNet, parallelised',
    src: 'Parallelizing Linear Transformers with the Delta Rule', arxiv: '2406.06484',
    bill: ['compute'], kv: { t: 'state' }, comp: 'lin', family: 'state',
    problem: 'The delta rule fixes linear attention\'s memory, but its write is sequential -- and a rule you cannot parallelise across the sequence is a rule you cannot train at scale.',
    mech: 'Reparameterise the sequence of rank-one delta updates as a matrix product, which allows a chunkwise parallel algorithm over sequence length.',
    buys: 'Made the delta rule trainable at real scale, and it beats Mamba on exactly the recall-intensive tasks fixed-state models were weakest on.',
    costs: 'More compute per token than plain linear attention, and the state is still fixed-size, so it has not escaped the fundamental compression.',
    pick: 'The fixed-state layer to reach for when recall matters and you are training at scale.',
    broke: 'The state could correct an association but never forget one.'
  },
  {
    id: 'gated-deltanet', date: '2024-12-09', name: 'Gated DeltaNet',
    src: 'Gated Delta Networks', arxiv: '2412.06464',
    bill: ['kv'], kv: { t: 'state' }, comp: 'lin', family: 'state', anchor: true,
    problem: 'A delta-rule state corrects what it holds but has no way to let anything decay, so stale associations accumulate indefinitely.',
    mech: 'Combine the delta write with a decay gate -- one gate controls how fast old memory fades, the other how strongly the new value is written.',
    buys: 'Correction and forgetting together. The strongest fixed-state layer of this generation on long-context recall.',
    costs: 'Still a fixed-size state, and still loses to real attention on exact retrieval from an arbitrary position. Two gates is two more things to tune.',
    pick: 'The D in a D/G hybrid schedule. This is what the notes\' DeltaNet layers are, and what Qwen3.6 ships in production.',
    broke: 'Nothing internally -- the remaining gap is structural, which is why every architecture using it also keeps some attention layers.'
  },
  {
    id: 'nsa', date: '2025-02-16', name: 'Native Sparse Attention',
    src: 'Native Sparse Attention', arxiv: '2502.11089',
    bill: ['compute'], kv: { t: 'compress', m: 16 }, comp: 'topk', family: 'sparse',
    problem: 'Two things kept breaking sparse attention: bolting it onto a dense-trained model degrades that model, and naive top-k saves nothing because you still have to score every key to find the top k.',
    mech: 'Three branches in parallel -- coarse compressed tokens, selected fine-grained blocks, and a sliding window -- with kernels written for the resulting access pattern, and the whole thing trained sparse from the start.',
    buys: 'Wall-clock speedup that survives contact with a real GPU, not just a FLOP count, while matching or beating full attention because the model was trained under this pattern rather than retrofitted to it.',
    costs: 'A commitment made on day one that you cannot cheaply undo. Kernel complexity is significant, and block granularity becomes a hyperparameter with real quality consequences.',
    pick: 'Training a new long-context model where you control the entire stack down to the kernels.',
    broke: 'Nothing -- but it set the standard that sparse patterns must be trained in, which is the assumption everything after it makes.'
  },
  {
    id: 'dsa', date: '2025-09-29', name: 'DeepSeek Sparse Attention',
    src: 'DeepSeek-V3.2-Exp release', arxiv: null,
    url: 'https://api-docs.deepseek.com/news/news250929/',
    bill: ['compute', 'kv'], kv: { t: 'compress', m: 8 }, comp: 'topk', family: 'sparse',
    problem: 'Section 7\'s catch, in production form: selection is only worth doing if proposing candidates is genuinely cheaper than scoring them all.',
    mech: 'A small lightning indexer keeps its own tiny key cache -- 128 per token against MLA\'s 512 -- scores incoming queries cheaply against it, and forwards only the top candidates to sparse MLA for the expensive read.',
    buys: 'The proposal step is small enough that the saving is real, and it drops long-context inference cost sharply without retraining the main attention path from scratch.',
    costs: 'The indexer can miss. It is another trained component that can be wrong, and it concentrates the entire approximation risk into one small model whose failures are hard to observe from the output.',
    pick: 'Long-context serving on a stack that already uses MLA.',
    broke: 'Nothing yet -- but the indexer is now a single point of quality failure that nobody has a good evaluation for.',
    note: 'Dated from the official release announcement rather than an arXiv submission.'
  },
  {
    id: 'drope', date: '2025-12-13', name: 'DroPE',
    src: 'Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings', arxiv: '2512.12167',
    bill: ['pos'], kv: { t: 'none' }, comp: 'none', family: 'position', anchor: true,
    problem: 'Every RoPE extension method is fighting RoPE\'s own inductive bias. The model has over-relied on explicit position, and stretching that signal is always a compromise.',
    mech: 'Use RoPE as a training-time scaffold -- it genuinely helps convergence -- then remove positional embeddings entirely after pretraining and briefly recalibrate at the original context length. Causal masking alone carries order after that.',
    buys: 'Zero-shot context extension with no long-context fine-tuning at all, for under one percent of the original pretraining budget, reported to beat established extension methods on LongBench and RULER.',
    costs: 'You must be able to run a recalibration phase, so it is not an inference-time switch. You give up explicit positional control entirely, and the approach inherits whatever weaknesses no-positional-encoding models have.',
    pick: 'Extending a pretrained model when a short recalibration is affordable but long-context training is not.',
    broke: 'Too new to say, which is the honest answer.',
    note: 'Two papers exist one capital letter apart. DRoPE (arXiv 2503.15029) is Directional Rotary Position Embedding for autonomous-driving trajectories and has nothing to do with context extension -- and it is the top search result for "DroPE". This card is the Sakana AI paper.'
  },
  {
    id: 'deepseek-v4-csa', date: '2026-04-26', name: 'Compressed + Heavily Compressed Attention',
    src: 'DeepSeek-V4', arxiv: '2606.19348',
    bill: ['compute', 'kv'], kv: { t: 'compress', m: 32 }, comp: 'topk', family: 'sparse', anchor: true,
    problem: 'At a million tokens, MLA plus sparsity together are still not enough. You have to cut how many positions are stored and how many of them get read.',
    mech: 'CSA consolidates every m tokens of KV cache into one entry and runs top-k sparse selection over those entries. HCA compresses far more aggressively but keeps dense attention over what remains. The two are interleaved through depth.',
    buys: 'At one million tokens, reported 27 percent of the per-token inference FLOPs and 10 percent of the KV cache relative to V3.2.',
    costs: 'Compression genuinely destroys token-level detail -- one summary now speaks for many tokens -- so recent detail needs explicit protection. Two mechanisms plus a schedule is a large tuning surface.',
    pick: 'Native million-token context where you are designing the architecture around that target from the start.',
    broke: 'Open. This is the current frontier of Road 2.'
  },
  {
    id: 'qwen36-hybrid', date: '2026-04-27', name: 'Qwen3.6 hybrid schedule',
    src: 'Qwen3.6 (35B-A3B) release', arxiv: null,
    bill: ['kv'], kv: { t: 'state' }, comp: 'lin', family: 'state',
    problem: 'If a depth schedule mixing fixed-state and attention layers is right, it should show up in more than one lab\'s architecture.',
    mech: 'A repeating cycle of three Gated DeltaNet blocks followed by one gated attention block, across 40 layers -- roughly three quarters of the network carrying a fixed-size recurrent state.',
    buys: 'Corroboration. A different lab, a different training stack, and the same three-to-one ratio of fixed-state to attention layers that section 13 describes.',
    costs: 'Same as any hybrid -- the ratio is a design choice nobody has cleanly ablated, and this release does not ablate it either.',
    pick: 'Not a mechanism to pick. Evidence that the schedule idea generalises beyond one model.',
    broke: 'Nothing. It raises the open question in the notes: the ratio is now convergent practice, still without a published ablation behind it.',
    note: 'Dated by model release rather than an arXiv submission.'
  }
];

const BILLS = {
  compute: { label: 'Compute (T²)', color: '#d95926' },
  kv: { label: 'KV cache (T)', color: '#3987e5' },
  pos: { label: 'Position / length', color: '#199e70' },
  sys: { label: 'Serving systems', color: '#9a6dd7' }
};

/* KV cache bytes for one user, using the section-10 formula:
   2 * layers * kv_heads * head_dim * T * batch * bytes_per_number
   Verified against the notes: the {t:'heads',h:8} model at T=32768 gives 6.44 GB. */
function kvBytes(model, T, batch) {
  const { layers, qHeads, headDim, bytes } = BASE;
  const per = (kvHeads, tokens) => 2 * layers * kvHeads * headDim * tokens * bytes;
  switch (model.t) {
    case 'mha': return per(qHeads, T) * batch;
    case 'heads': return per(model.h, T) * batch;
    case 'none': return per(BASE.kvHeadsGQA, T) * batch;
    case 'window': return per(BASE.kvHeadsGQA, Math.min(T, model.w)) * batch;
    case 'compress': return per(BASE.kvHeadsGQA, Math.ceil(T / model.m)) * batch;
    case 'latent': return layers * model.dim * T * bytes * batch;
    // A linear-attention / DeltaNet state is S = sum of v k^T -- a head_dim x head_dim
    // MATRIX per head, not a vector. Note there is no T term: this is the same size after
    // ten tokens and after a million, which is the entire point of the fixed-state family.
    case 'state': return layers * BASE.qHeads * headDim * headDim * bytes * batch;
    default: return per(BASE.kvHeadsGQA, T) * batch;
  }
}

/* ---------------------------------------------------------------------------
 * Per-mechanism visuals.
 *
 * Thirty bespoke animations would be thirty chances to draw something decorative
 * that does not match the mathematics. Instead every mechanism declares one of six
 * primitives, each of which renders from the actual rule rather than from a picture
 * of it -- the pattern grids run the real selection function, the position curves
 * evaluate the real positional signal, the state panels run the real write rule.
 *
 *   pattern  which keys a query actually reads   (the sparse / window / compress family)
 *   pos      positional signal vs distance, with the training boundary marked
 *   heads    query heads -> stored K/V heads wiring
 *   state    the fixed-size state being written and read back
 *   sys      a systems-layer change with no effect on the mathematics
 *   sched    fixed-state vs attention layers through depth
 */
const VIZ = {
  'learned-absolute':      { k: 'pos',     fn: 'table' },
  'standard-attention':    { k: 'pattern', fn: 'causal' },
  'sinusoidal':            { k: 'pos',     fn: 'sin' },
  'transformer-xl':        { k: 'pattern', fn: 'segment', seg: 9 },
  'sparse-transformer':    { k: 'pattern', fn: 'strided', stride: 5, local: 3 },
  't5-relative-bias':      { k: 'pos',     fn: 'bucket' },
  'mqa':                   { k: 'heads',   q: 8, kv: 1 },
  'longformer':            { k: 'pattern', fn: 'global', w: 5, g: 2 },
  'linear-attention':      { k: 'state',   fn: 'add' },
  'delta-rule':            { k: 'state',   fn: 'delta' },
  'rope':                  { k: 'pos',     fn: 'rope' },
  'alibi':                 { k: 'pos',     fn: 'alibi' },
  'flashattention':        { k: 'sys',     fn: 'flash' },
  'gqa':                   { k: 'heads',   q: 8, kv: 2 },
  'position-interpolation':{ k: 'pos',     fn: 'pi' },
  'ntk-aware':             { k: 'pos',     fn: 'ntk' },
  'yarn':                  { k: 'pos',     fn: 'yarn' },
  'paged-attention':       { k: 'sys',     fn: 'paged' },
  'attention-sinks':       { k: 'pattern', fn: 'sink', w: 6, s: 2 },
  'mistral-swa':           { k: 'pattern', fn: 'window', w: 6 },
  'mamba':                 { k: 'state',   fn: 'selective' },
  'mla':                   { k: 'heads',   q: 8, kv: 'latent' },
  'ssd':                   { k: 'pattern', fn: 'decay' },
  'parallel-deltanet':     { k: 'state',   fn: 'chunk' },
  'gated-deltanet':        { k: 'state',   fn: 'gated' },
  'nsa':                   { k: 'pattern', fn: 'nsa' },
  'dsa':                   { k: 'pattern', fn: 'topk', top: 5 },
  'drope':                 { k: 'pos',     fn: 'drope' },
  'deepseek-v4-csa':       { k: 'pattern', fn: 'block', m: 4, top: 2 },
  'qwen36-hybrid':         { k: 'sched',   motif: 'DDDG' }
};

/* Which keys query i actually reads. Returns 0 skipped, 1 read exactly,
   2 read through a compressed/summarised entry. j > i is always the future. */
function readMask(v, N) {
  const M = [];
  const score = (i, j) => { const x = Math.sin(i * 12.9898 + j * 78.233) * 43758.5453; return x - Math.floor(x); };
  for (let i = 0; i < N; i++) {
    const row = new Array(N).fill(0);
    const allowed = [];
    for (let j = 0; j <= i; j++) allowed.push(j);
    switch (v.fn) {
      case 'causal': allowed.forEach(j => row[j] = 1); break;
      case 'window': allowed.forEach(j => { if (j > i - v.w) row[j] = 1; }); break;
      case 'sink': allowed.forEach(j => { if (j < v.s || j > i - v.w) row[j] = 1; }); break;
      case 'global': allowed.forEach(j => { if (j < v.g || j > i - v.w) row[j] = 1; }); break;
      case 'strided': allowed.forEach(j => { if (j > i - v.local || (i - j) % v.stride === 0) row[j] = 1; }); break;
      case 'segment': {
        const seg = Math.floor(i / v.seg);
        allowed.forEach(j => { const sj = Math.floor(j / v.seg); row[j] = sj === seg ? 1 : sj === seg - 1 ? 2 : 0; });
        break;
      }
      // returns the ACTUAL weight, not a flag -- the renderer shades by it, because
      // the decay is the mechanism and a flat band would hide it.
      case 'decay': allowed.forEach(j => { const w = Math.pow(0.82, i - j); row[j] = w > 0.02 ? w : 0; }); break;
      case 'topk': {
        const ranked = allowed.slice().sort((a, b) => score(i, b) - score(i, a)).slice(0, v.top);
        ranked.forEach(j => row[j] = 1);
        if (i > 0) row[i] = 1;
        break;
      }
      case 'block': {
        const nb = Math.floor(i / v.m) + 1, blocks = [];
        for (let b = 0; b < nb; b++) blocks.push(b);
        const keep = blocks.slice().sort((a, b) => score(i, b) - score(i, a)).slice(0, v.top);
        allowed.forEach(j => { if (keep.indexOf(Math.floor(j / v.m)) >= 0) row[j] = 2; });
        allowed.forEach(j => { if (j > i - 3) row[j] = 1; });
        break;
      }
      case 'nsa': {
        allowed.forEach(j => { if (j > i - 4) row[j] = 1; });                       // window branch
        allowed.forEach(j => { if (Math.floor(j / 4) % 3 === 0 && !row[j]) row[j] = 2; }); // compressed branch
        const ranked = allowed.filter(j => !row[j]).sort((a, b) => score(i, b) - score(i, a)).slice(0, 3);
        ranked.forEach(j => row[j] = 1);                                            // selected branch
        break;
      }
      default: allowed.forEach(j => row[j] = 1);
    }
    M.push(row);
  }
  return M;
}

/* Positional signal as a function of distance, plus where training stopped.
   Returns {y:[...], train:idx, wall:bool, note}. Shape is what matters here, not units. */
function posCurve(fn, N, train) {
  const y = [], TAU = Math.PI * 2;
  for (let d = 0; d < N; d++) {
    let v;
    switch (fn) {
      case 'table':  v = d < train ? 0.5 + 0.40 * Math.sin(d * 0.55) : NaN; break;
      case 'sin':    v = 0.5 + 0.40 * Math.sin(d / 3.1); break;
      case 'rope':   v = 0.5 + 0.40 * Math.cos(d / 3.1); break;
      case 'alibi':  v = Math.max(0, 0.95 - d * (0.95 / N)); break;
      case 'bucket': { const b = Math.min(Math.floor(Math.log2(1 + d) * 2), 7); v = 0.9 - b * 0.11; break; }
      case 'pi':     v = 0.5 + 0.40 * Math.cos((d * train / N) / 3.1); break;
      case 'ntk':    v = 0.5 + 0.40 * Math.cos(d / (3.1 * (1 + 1.6 * d / N))); break;
      case 'yarn':   v = 0.5 + 0.40 * Math.cos(d / (3.1 * (d < train ? 1 : 1 + 1.4 * (d - train) / N))); break;
      case 'drope':  v = 0.5; break;
      default:       v = 0.5;
    }
    y.push(v);
  }
  const NOTE = {
    table:  ['Hard wall', 'The table has no row past its last one. Not degraded — undefined.'],
    sin:    ['Defined, untrained', 'The function keeps producing values past the training length. The model never learned to read them.'],
    rope:   ['Relative, still untrained', 'Depends only on i−j, so shifting both tokens changes nothing. But the frequencies were calibrated on the trained range.'],
    alibi:  ['Extrapolates cleanly', 'A fixed decay with no wall — and a recency prior the model cannot overrule.'],
    bucket: ['Saturates', 'Past the last bucket every distance looks identical.'],
    pi:     ['Squeezed to fit', 'All frequencies compressed equally — reach bought by blurring local detail.'],
    ntk:    ['Stretched unevenly', 'Low frequencies stretch, high frequencies barely move, so local detail survives.'],
    yarn:   ['Banded', 'Frequencies treated differently depending on whether their wavelength fit the trained window.'],
    drope:  ['Removed entirely', 'No positional signal at all. Order comes from the causal mask alone.']
  }[fn] || ['', ''];
  return { y, train, wall: fn === 'table', title: NOTE[0], note: NOTE[1] };
}
