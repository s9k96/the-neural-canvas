# S6 Evidence — V5 Training Data Execution System

**Result: PASS** — 9/9 requirements passed.

Every row below was recomputed by `tds/evidence.py` from the files in `submission_artifacts/`, not from the state of the run that produced them.

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | `manifests/tokenizer_manifest.json + manifests/manifest_validation.json` — 35 shard manifests carry the frozen tokenizer hash; 35/35 manifests revalidated from shard bytes |
| Evaluation firewall | PASS | `manifests/eval_registry.json + ledgers/consumption.jsonl` — 3 block events; 5 never-train shards, 0 of them in 21 consumed shard ids; 6 shards rejected at the gate |
| Packing correctness | PASS | `manifests/packed_batch_report.json` — 380 packed batches; utilization recomputed from placements = 0.9818; 500 batches mask-audited |
| Mixture compliance | PASS | `manifests/mixture_schedule.json + ledgers/mixture_compliance.json` — planned vs actual max drift 0.0001; 8 floor windows checked, 0 violations |
| OPUS audit trail | PASS | `ledgers/opus_decisions.jsonl` — 1520 candidate decisions: accepted=925, deferred=413, protected=14, rejected=168; 640 decision ids referenced by consumption events all resolve |
| Crash recovery | PASS | `checkpoints/resume_proof.json + checkpoints/checkpoints_index.json` — expected batch 40 matched on resume; effective stream = 320 events over steps [0, 79], 0 gaps, 0 duplicates, 60 superseded |
| Replay | PASS | `checkpoints/replay_proof.json + checkpoints/fork_proof.json` — 80/80 replayed batch hashes identical to the original run; fork fork-1 diverges at step 40 and re-runs identically |
| Learning trace | PASS | `ledgers/learning_ledger.jsonl + ledgers/token_trace.jsonl + ledgers/token_stats.json` — 21 shard report cards + 7 lane roll-ups, every key present in the consumption ledger; 13403 token rows (3872 full-tier) |
| Throughput | PASS | `performance.json + ledgers/consumption.jsonl` — useful 153880 / 45.01s = 3418.5 tok/s (reported 3418.526); packing utilization recomputed 0.9818 (reported 0.981774) |

## How each row was verified

### Tokenizer integrity — PASS
*Rubric area:* Shards, manifests and tokenizer integrity  
*Evidence:* `manifests/tokenizer_manifest.json + manifests/manifest_validation.json` at `$.frozen_sha256 ; $.rows[*].checks.content_hash_matches`  
*Method:* re-hashed the tokenizer file and recomputed every shard content hash

- PASS — `frozen_constant_matches_manifest`
- PASS — `file_on_disk_matches_frozen`
- PASS — `every_shard_carries_frozen_hash`
- PASS — `all_manifests_validated`
- PASS — `content_hash_recomputed_for_every_shard`
- PASS — `vocab_remap_recorded`

### Evaluation firewall — PASS
*Rubric area:* Evaluation and validation firewall  
*Evidence:* `manifests/eval_registry.json + ledgers/consumption.jsonl` at `$.blocked_events ; $.never_train_shard_ids`  
*Method:* intersected the never-train registry with every shard id in the ledger

- PASS — `registry_populated`
- PASS — `never_train_shards_registered`
- PASS — `eval_shard_blocked_event_present`
- PASS — `contamination_blocked_event_present`
- PASS — `no_never_train_shard_consumed`
- PASS — `no_validation_shard_consumed`
- PASS — `rejected_shards_kept_reasons`
- PASS — `access_logged`

### Packing correctness — PASS
*Rubric area:* Packing, masks and batch correctness  
*Evidence:* `manifests/packed_batch_report.json` at `$.batches[*] ; $.mask_invariants`  
*Method:* re-added every placement span and re-derived utilization per batch

- PASS — `batches_reported`
- PASS — `position_arithmetic_recomputes`
- PASS — `utilization_in_range`
- PASS — `placements_sum_to_used_positions`
- PASS — `loss_never_on_padding`
- PASS — `loss_never_on_context`
- PASS — `attention_never_crosses_segments`
- PASS — `position_ids_restart_per_segment`
- PASS — `policy_comparison_present`

### Mixture compliance — PASS
*Rubric area:* Mixture schedule, protected floors and OPUS  
*Evidence:* `manifests/mixture_schedule.json + ledgers/mixture_compliance.json` at `$.planned_allocator_shares vs recomputed actual ; $.floor_windows[*]`  
*Method:* recounted lane shares from every effective ledger event

- PASS — `schedule_compiled`
- PASS — `per_step_quotas_present`
- PASS — `planned_vs_actual_within_tolerance`
- PASS — `integrated_stages_track_s5_headline`
- PASS — `protected_floors_hold_every_window`
- PASS — `floors_declared`
- PASS — `anneal_reserve_declared`
- PASS — `anneal_reserve_unused_before_anneal`
- PASS — `supply_reconciled`

### OPUS audit trail — PASS
*Rubric area:* Mixture schedule, protected floors and OPUS  
*Evidence:* `ledgers/opus_decisions.jsonl` at `$[*].status ; $[*].protected_floor_override`  
*Method:* joined every consumption event's opus_decision_id against the decision log

- PASS — `decisions_recorded`
- PASS — `all_11_fields_present`
- PASS — `four_ledgers_populated`
- PASS — `rejections_carry_reasons`
- PASS — `floor_override_only_on_protected_lanes`
- PASS — `every_consumed_batch_traces_to_a_decision`
- PASS — `scores_are_real_numbers`
- PASS — `scoring_checkpoint_recorded`

### Crash recovery — PASS
*Rubric area:* Checkpoint, crash, resume, replay and fork  
*Evidence:* `checkpoints/resume_proof.json + checkpoints/checkpoints_index.json` at `$.next_batch_matched ; $.expected_next_batch.plan_digest`  
*Method:* rebuilt the effective stream from the ledger and diffed it against the checkpoint's pre-crash prediction

- PASS — `proof_hashes_match_ledger_attempt1`
- PASS — `proof_hashes_match_ledger_attempt2`
- PASS — `ledger_attempts_agree`
- PASS — `crash_simulated`
- PASS — `resumed_from_checkpoint`
- PASS — `expected_next_batch_predicted_before_crash`
- PASS — `resumed_batch_matches_expected`
- PASS — `resumed_plan_digest_matches`
- PASS — `no_missing_steps`
- PASS — `no_duplicate_batches`
- PASS — `uniform_microbatches_per_step`
- PASS — `superseded_events_recorded`
- PASS — `checkpoint_offsets_match_ledger`

### Replay — PASS
*Rubric area:* Checkpoint, crash, resume, replay and fork  
*Evidence:* `checkpoints/replay_proof.json + checkpoints/fork_proof.json` at `$.comparisons[*].hash_match ; $.diverged`  
*Method:* re-ran the planner from the older checkpoint and compared hashes, span ids and loss-mask hashes against the original ledger events

- PASS — `replay_ran`
- PASS — `all_batch_hashes_match`
- PASS — `proof_hashes_match_ledger`
- PASS — `fork_proof_hashes_match_ledger`
- PASS — `all_token_spans_match`
- PASS — `all_loss_mask_hashes_match`
- PASS — `replayed_from_older_checkpoint`
- PASS — `fork_recorded`
- PASS — `fork_diverges`
- PASS — `fork_reproducible`
- PASS — `fork_divergence_point_recorded`

### Learning trace — PASS
*Rubric area:* Consumption and learning ledgers  
*Evidence:* `ledgers/learning_ledger.jsonl + ledgers/token_trace.jsonl + ledgers/token_stats.json` at `$[*].loss_delta_before_after ; $[*].shard_id`  
*Method:* joined every learning-ledger key and token-trace row back onto the shard ids in the consumption ledger

- PASS — `learning_ledger_written`
- PASS — `all_11_fields_present`
- PASS — `every_row_traces_to_consumed_shard`
- PASS — `token_trace_written`
- PASS — `token_trace_rows_reference_real_shards`
- PASS — `full_tier_present`
- PASS — `aggregate_tier_present`
- PASS — `losses_are_finite`
- PASS — `usefulness_classified`

### Throughput — PASS
*Rubric area:* Throughput and packing efficiency  
*Evidence:* `performance.json + ledgers/consumption.jsonl` at `$.metrics.useful_loss_bearing_tokens_per_sec ; $.metrics.packing_utilization`  
*Method:* re-summed loss-bearing positions from the effective ledger stream and divided by the recorded wall time

- PASS — `all_10_metrics_present`
- PASS — `accepted_tokens_per_sec_recomputes`
- PASS — `useful_tokens_per_sec_recomputes`
- PASS — `packing_utilization_recomputes`
- PASS — `raw_ge_accepted_ge_useful`
- PASS — `discarded_work_accounted`
- PASS — `wall_time_positive`
- PASS — `rejection_rate_reported_per_lane`
- PASS — `cache_metrics_measured`
