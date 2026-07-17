Axiom

Courses
ERA V5
Session 3: Data Collection & Sourcing

Search...


Mark Complete
Exit
Session 3: Data Collection and Sourcing
1. What this session is
The first two sessions covered the model itself: the neuron, the network, and the tokenizer that converts language into tokens. This session moves to the data used to train that model.

By 2026, serious labs use broadly similar transformer architectures. The corpus creates much of the remaining difference. Data is the model: it determines what the model sees, what it learns, which capabilities emerge, and which weaknesses remain. Every later stage inherits the decisions made during corpus construction.

ERA V4 gives us a concrete reference. We assembled roughly 1.15 trillion tokens and trained a 120B mixture-of-experts model, along with 9B, 5B, and 2B variants. The capstone run required about sixty-seven days on one eight-GPU node and would have cost close to $100,000 without spot pricing.

That corpus was large for a student-built effort. It remains small beside current frontier runs. Llama 4 used more than 30 trillion tokens. Qwen3 used 36 trillion tokens across 119 languages. Reaching that scale requires more than compute. It requires collection, extraction, cleaning, filtering, deduplication, balancing, provenance tracking, and evaluation.

This session covers five questions:

What does a dataset look like at each training stage?
How much data is required?
Which data creates which capability?
How should low-resource Indic data be protected and expanded?
How do we collect data without losing quality, provenance, or evaluation integrity?
2. A first look at the corpus we built
Start with the V4 corpus. It shows that a training corpus is an engineered system, not a directory of downloaded text.

The final corpus contained 1,118 billion tokens across 33,353 shards. It had three tiers:

Main pretraining pools, D1 to D4. These moved from high-quality web data to broader web data, code, science, and mathematics. OPUS scored candidate batches and retained roughly forty percent.
Always-on data. This supplied eight percent of every batch, independent of the selector. It protected Indic data and benchmark training splits that the selector would otherwise under-sample.
Golden proxy data. Benchmark test material never trained the model. It only supplied the direction used by the selector.

The LightningLM V0.1 corpus divided into its three operational tiers.

3. What a dataset is for
A dataset is defined by its training objective. The unit of data changes across the model lifecycle.

Pretraining: a document. The model learns language, knowledge, code, and statistical structure through next-token prediction.
Mid-training or annealing: a smaller reserve of high-quality documents, usually concentrated in code, mathematics, reasoning, or target languages.
Supervised fine-tuning: an instruction-response pair. The model learns how to answer rather than only continue text.
Preference optimisation: a prompt with chosen and rejected responses. The model learns which valid-looking answer is preferred.
Reinforcement learning: a prompt, generated trajectory, and reward or verifier. The model is optimised against measurable outcomes.
Across these stages, required data volume falls by roughly a million times. Quality requirements move in the opposite direction. Pretraining can tolerate some noise at massive scale. A faulty reinforcement-learning verifier can directly teach the wrong behaviour.


Select a modality, then move through pretraining, annealing, SFT, preference optimisation, and RL. The unit changes from a document to an instruction-response pair, a chosen-rejected comparison, and finally a prompt with a verifier. The volume bar collapses while the noise-tolerance bar inverts. The held-out rail never trains the model.

The held-out rail is a separate data system. It exists to measure the model. Once held-out data enters training, its score can no longer be trusted.

4. How much data, and the wall ahead
The first planning problem is the token budget.

Chinchilla scaling places the compute-optimal point near twenty training tokens per parameter. Modern models commonly train far beyond that point. Additional tokens provide diminishing returns during training, but they allow a smaller model to become stronger and remain cheaper during every future inference request.


Thirty-two models plotted by parameter count and training tokens. The Chinchilla line marks twenty tokens per parameter. Move the parameter and token dials, drag the V5 marker across the plane, and hover over any model to inspect its budget. A dashed ring marks token counts that were not officially disclosed.

A large token target immediately creates a supply problem. One option is to repeat the same data.


Unique tokens versus repeated passes. Repetition remains nearly free up to roughly four passes, then loses value and approaches wasted compute by around sixteen. The control makes the marginal value of each additional pass visible.

Repetition matters because frontier token budgets have grown faster than the supply of fresh, high-quality human text.


Frontier pretraining budgets on a log scale from GPT-3 through Llama 4 and Qwen3, compared with the previous campaign and the V5 target. V5 is shown as a 10 to 30 trillion-token range because the final value depends on model size and on how much synthetic and multilingual data clears the quality bar.

The public web is finite. Once the usable portion has been collected, filtered, and deduplicated, more crawling does not create equivalent new information.


Frontier demand and usable public-text supply projected into the early 2030s. Toggle synthetic generation, multilingual collection, and licensed sources to move the crossing point. Every crossing date is a projection, not a certainty.

Epoch AI estimates the effective stock of public human text at roughly 300 trillion tokens, with full utilisation projected between 2026 and 2032. The exact date is uncertain. The engineering consequence is already clear: future corpora require synthetic generation, new languages, licensed sources, or all three.

5. Which data buys which capability
Corpus design should begin with the target capability and benchmark. Work backwards from the required behaviour to the data most likely to produce it.

Mathematics requires mathematical text, worked solutions, proofs, code, and verifiable problems. Coding requires repositories, documentation, tests, issue discussions, and execution feedback. Multilingual ability requires native text, parallel text, domain diversity, and a tokenizer that does not waste the token budget.


Data sources on the left and benchmarks on the right, joined by weighted links. Select a source to show every benchmark it feeds, or select a benchmark to trace the data that improves it. Each link carries the published delta behind it.

Code data also improves tasks that contain no code. It teaches exact structure, decomposition, variable tracking, symbolic manipulation, and long dependency chains. The share still matters. Excessive code can reduce performance elsewhere.


The first tab compares training with and without code on non-code benchmarks, including the gains and the cost of an excessive code share. The second animates the DeepSeekMath mining loop: a mathematics seed trains a classifier, the classifier extracts more mathematics from the web, and repeated mining produces 120 billion tokens that take a 7B model to competition-level scores.

The DeepSeekMath loop is important because it converts a small trusted seed into a domain-scale corpus. The same pattern applies when no ready-made high-quality corpus exists for a language or domain.

6. Quality over quantity
Raw token count is not the objective. Useful learning signal is.

A smaller, well-filtered corpus can outperform a much larger raw corpus at the same training budget. Low-quality text consumes compute while contributing weak, duplicated, malformed, or misleading signal.


A fixed-budget comparison between a large raw corpus and a smaller filtered corpus. The filtered corpus wins on knowledge benchmarks while using less data and compute. Every figure is labelled as a published result from FineWeb-Edu or DataComp on the authors' own evaluations.

Frontier-scale filtering usually follows a two-stage pattern:

A strong model labels a manageable sample for educational value, correctness, structure, or domain relevance.
A cheap classifier learns those labels and scores the full corpus.

A large model scores a sample for educational value, a lightweight classifier learns those labels, and the classifier filters the full corpus. Move the threshold to watch retained volume fall and average quality rise, with kept and discarded token counts computed live.

The classifier is part of the data recipe. Its labels, training distribution, threshold, language coverage, and failure modes must be tested like any other model component.

7. Deduplication and synthetic data
Deduplication removes repeated learning signal, benchmark leakage, and overrepresented sources. Aggressive deduplication can also remove useful text.


Toggle between per-snapshot and global deduplication. Global deduplication removes more tokens but scores worse because it can discard higher-quality material from older crawls. This is why FineWeb retained the gentler per-snapshot method.

The deduplication scope must therefore be validated experimentally. More removal is not automatically better.

Synthetic data expands a limited human corpus. It can rewrite weak documents, generate explanations, translate content, produce exercises, add reasoning traces, or create verifiable code and mathematics.


Control the real-to-synthetic mix using the Nemotron-CC pattern. Moderate synthetic augmentation improves quality. Beyond that range, quality falls as the corpus increasingly feeds on model output, exposing the model-collapse risk.

Synthetic data is an amplifier. Its value depends on the teacher model, prompt design, verification, diversity, and the quality of the real seed corpus.

8. The sovereign thread: Indic data
V5 is intended to build strong capability in languages that remain poorly represented on the global web. This requires native collection, language-aware filtering, and explicit protection during training.

Headline multilingual token counts can be misleading. Many corpora contain large volumes of machine-translated or otherwise synthetic text. The verified human-origin split gives a better measure of actual language coverage.


Sangraha totals split into verified human-origin data and everything else. Switch to verified-only view to see Telugu fall from 16.3 billion tokens to 3.7 billion and Odia from 12.5 billion to 1.2 billion. The English comparison places these slivers beside CulturaX's 2,847 billion English tokens.

The scarcity becomes worse when an English-trained quality filter is applied. Such filters often penalise local writing styles, uncommon domains, code-mixed text, and Brahmic-script structure. A strict filter can improve English data while deleting the languages the model is meant to learn.


Increase the strictness of an English-tuned filter and Indic-script content disappears much faster than English content. The same failure caused the V4 selector to undervalue Indic data. The script-aware fix prevents that erasure.

This explains the V4 always-on channel. The English-heavy selector undervalued Indic data, so Indic samples were removed from selector control and guaranteed a fixed share of every batch. Low-resource data protection is a core part of the training design.

9. Provenance, openness, and licence
Every source must answer three questions:

Where did the data come from?
What transformations were applied?
What rights and obligations govern its use and release?
Open weights and open training data are separate properties. A model can publish its weights while disclosing almost nothing about its corpus.


Models positioned by weight openness and data openness. Most models called open publish weights while keeping the corpus secret. Select a model to inspect exactly what it released.

Licences also combine across a mixture. Attribution, share-alike, non-commercial restrictions, and unknown provenance can affect the usability of the final corpus and model.


Assemble a corpus from permissive, attribution, share-alike, and unknown sources. Adding one source with unknown provenance flags the full mixture as unusable, showing why provenance must be tracked from the first day.

Provenance must be captured during collection. Reconstructing it after files have been merged, cleaned, shuffled, and sharded is unreliable.

10. From raw pages to training text
Collection produces raw material. Extraction converts that material into text suitable for training.

A naive HTML strip leaves navigation, cookie banners, repeated menus, hidden text, malformed Unicode, entities, boilerplate, and control characters. These become tokens and consume the training budget.


The same raw page extracted two ways. The naive method leaves navigation, cookie banners, HTML entities, and invisible control characters. Switch to the browser-based cleaning routine and watch the garbage-token count fall to zero.

The second operation is the anneal. A small reserve of the best data is held back and used during the learning-rate cooldown near the end of pretraining.


General-web pretraining followed by a small mathematics-and-code reserve during the learning-rate cooldown. Compose the anneal mixture and watch the OLMo 2 result move grade-school mathematics from 24 percent to 67 percent using a reserve that is tiny beside the full run.

The anneal corpus must be planned before training begins. Spending all high-quality data in the general pool removes the ability to concentrate it at the end.

11. Testing a recipe before we scale it
A trillion-token decision should not be based on intuition.

Candidate mixtures, filters, thresholds, deduplication methods, and synthetic-data strategies should first be tested on smaller proxy models. The proxy run does not need to reproduce the final score. It needs to rank competing recipes reliably.


Several data recipes run on small proxy models, are ranked by result, and only the winner is scaled. Run the ablations to see both the final gain and the compute saved by never testing losing recipes at full scale.

Every important data decision should become an ablation. Without an ablation, a corpus change remains an opinion.

12. Keeping the evaluation honest
Training data and evaluation data require a hard boundary.

Test leakage raises the reported score without creating equivalent general capability. It can enter through copied benchmark pages, solution repositories, synthetic examples based on test questions, instruction datasets, or careless mixing of benchmark splits.


Three separated tiers: training pools, benchmark training splits permitted for format familiarity, and a golden proxy built from test material that never enters a batch. Deliberately leak the test set to create a dishonest score jump, then enforce the firewall to recover the real score.

Evaluation integrity requires source-level exclusion, exact and fuzzy matching, semantic contamination checks, and immutable held-out sets. Once a test item trains the model, that item is no longer a valid test.

13. The minor thread: the tokenizer as the first data decision
The tokenizer is the first data-allocation mechanism.

Tokenizer fertility is the number of tokens required to encode a piece of text. A language with higher fertility consumes more context length and more training tokens for the same semantic content. Under a fixed token budget, it receives fewer sentences, fewer documents, and less knowledge.


The same sentence tokenised in English and several Indian languages, with the token cost shown for each. The fertility control compares how many words fit into a fixed budget under a generic English-centric tokenizer and an Indic-aware tokenizer.

Session 2 showed how the vocabulary is created. This session shows what that vocabulary costs. Tokenizer design and corpus design are one system.

14. The assignment
Assume that you have to train a 40B model, which is as good as Gemma 4, is really good in coding, agentic work, Indic languages and is India first (views the world from Indian perspective). Do your research with your agents and decide:

how your data would look like, what would you collect and why (for pre-training, post-training, RL/alignment)
how would you clean you data for the objectives defined
how would you test your models against the objective
what fertility would you target for the different languages (which languages would you focus on), coding, science, math and agentic tasks, and based on these number, what would be your tokenizer size (token vocab size)
Once you're sure of your number, work with your agent to write a report, upload it on netlify and share back with us. Our evaluation would be based on how much have you thought through. Longer submissions will result is lower scores.

Transcript

Video
Studio


GMeet


Previous
Assignment Solution for S2

