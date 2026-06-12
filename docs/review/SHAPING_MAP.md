# Reward-shaping / quality-score map (env-axis review input, 2026-06-12)

Agent-produced map, lightly edited. Axis under judgment: *the environment's
only job is selection pressure whose sole deliverable is a portable,
perception-coupled NN policy.* Companion file: `ACTION_MAP.md` (action
subsystem). Synthesis: `../ENV_AXIS_REVIEW.md`.

Architectural fact that frames everything below: there are **four distinct
sinks** for "shaping" signal, and only one of them is selection.

| Sink | Where | Selection-relevant? |
|---|---|---|
| Energy/health deltas | action handlers, upkeep | YES — the only true selection currency (death lines at `simulation.py:1155-1159`, spawn thresholds) |
| Valence → within-lifetime weight updates | `simulation.py:370-376` → `controller.py:409-416` | Indirect — shapes what a controller learns, not who persists |
| outcome_targets / feedback → prediction heads + observation channels | `simulation.py:377-383`, `individuals.py:216-292` | Indirect — supervision targets and self-perception inputs |
| success_profile / counters → checkpoint score + champions | `individuals.py:294-299`, `simulation.py:3245-3363` | NOT selection. Pure extraction-side ranking (observer is descriptive-only, `observer.py:12-14`). One leak: `child_count` feeds `_partner_score` (`simulation.py:2804`) |

## 1. Shaping-term table

### 1a. Valence terms (the per-tick scalar that drives policy learning, `simulation.py:369-376`)

| Term | Line | Reads | Rewards | Magnitude (vs typical valence 0.1-0.7) | Verdict |
|---|---|---|---|---|---|
| `valence_energy * energy_delta/10` | sim:371 | post-action energy delta | net energy gain | eat gain 1-7/tick → 0.02-0.6 (coef 0.20-0.85, params.py:189) | **SERVES** — energy is the honest signal; perceptible via observation feature 0 (sim:1291) |
| `valence_health * health_delta*4` | sim:372 | health delta | avoiding/repairing damage | small, ±0.1 typical (coef 0.15-0.65) | SERVES (mild) |
| `-valence_damage * damage*4` | sim:373 | damage | penalizes harm | comparable to health term (coef 0.35-0.95) | SERVES (mild) |
| `valence_spawn * spawning` | sim:374 | `feedback["spawning"]` = 1.0 per birth (sim:2685, 2780-2781) | spawning | up to 0.55 per birth — the largest single-event valence | **DILUTES** as long as spawn *timing* is carried by harness channels (drive injection, intent window, §2); the controller gets a big learning pulse for an event it did not have to perceive its way into |
| `valence_social * social` | sim:375 | `feedback["social"]` 0.04-0.5/event | signaling, collaborating, observing | coef 0.00-0.35 → ≤0.15, usually ≤0.05 | DILUTES (variance, no perception demand) — near-dead at typical coefs |
| `movement_hazard = damage*4 if move` | sim:369, 382 | move damage | prediction target only (`hazard` head), not valence | supervision-only | SERVES weakly (prediction supervision is perception-coupled) |
| `tool` feedback → `outcome_targets["tool"]` + observation | sim:381, individuals.py:225,250,289 | tool/craft/build events | prediction target + self-perception channel; **never enters valence or energy** | 0.05-1.0/event | DEAD SURFACE for selection; mild SERVE as a prediction-supervision target |

### 1b. Feedback-channel writers (what fills `spawning`/`social`/`tool`)

| Event | Line | Writes | Magnitude |
|---|---|---|---|
| move success w/ helpers | sim:1550-1551 | social +support*0.28, tool +support*0.10 | ≤0.23/0.08 |
| move fail w/ helpers | sim:1580 | social +support*0.12 | ≤0.10 |
| craft fail | sim:1828 | tool +skill_gain | ~0.005-0.02 |
| craft success | sim:1850-1851 | social +0.06; tool +0.30+0.30·method_quality | ≤0.6 |
| build fail | sim:1941 | tool +0.04+lost*0.01+support*0.04 | ~0.05-0.15 |
| build success | sim:1984-1987 | social +0.10(+support*0.20); tool +0.75+support*0.14 | ~0.9 |
| use_tool success | sim:2053-2056 | social +0.2(+support*0.18); tool +1.0+support*0.12 | ~1.1 (largest tool pulse) |
| use_tool fail | sim:2109 | tool +0.05+support*0.02 | tiny |
| causal step | sim:2229 | tool +0.10 | tiny |
| causal unlock | sim:2253 | tool +0.70 | medium |
| signal | sim:2383 | social +intensity*0.1 | ≤0.1 |
| coordinate | sim:2402 | social +intensity*0.12 | ≤0.08 |
| mark | sim:2455 | social +0.04+intensity*0.08+clarity*0.04 | ≤0.15 |
| clone birth | sim:2685 | spawning +1.0 | 1.0 |
| combine birth (each parent) | sim:2780-2781 | spawning +1.0 | 1.0 |
| read mark | sim:3034-3035 | social +0.05+fidelity*0.04; tool +min(0.12, gain*4) | ≤0.21 |
| observe demo | sim:3114-3115 | social +0.2/0.05; tool +0.10/0.03 | small |

### 1c. Energy-side shaping beyond raw foraging (the actual selection currency)

| Term | Line | Reads | Magnitude vs upkeep (~0.06-0.16/tick, individuals.py:155-190) |
|---|---|---|---|
| `_tool_effect` gains | sim:2152-2197 | place resources, affordance, skill, params | 0.5-12 energy/use — dominant income for tool users |
| causal step gain `0.05+planning*0.08` | sim:2243 | planning | trivial |
| causal unlock: release ≤ `3+13c+6p+seq*2+3s` to place, actor share `release*(0.30+sensor*0.10+pred*0.12+planning*0.10)` | sim:2248-2250, 2293 | competence, planning, params | up to ~12 energy direct + replenishes place — the biggest single perception-flavored payout |
| drain transfer | sim:2324-2373 | weakest co-located target | 30-75% of the target's energy on deactivation |
| death recycling: `+2.0 + 0.35*energy` residue to place | sim:3146 | — | persistent food-base subsidy |
| repair: energy→health when energy > 20× upkeep | individuals.py:324-327 | own energy | small drain |
| `planning` (`_interaction_control`) multiplier | sim:458-471 | prediction_error_profile, event_memory, skill breadth, place_memory | adds up to +0.26 use_tool chance (sim:2039), +0.30 causal margin (2218), +0.16 build chance (1932), cost discounts everywhere |

**Axis verdict on `planning`:** half-and-half. It rewards low prediction
error (which lives in the extracted NN's heads, perception-coupled), but the
*conversion* of that into success-chance bonuses is scripted harness code.
The extracted policy's realized competence partly resides in an amplifier
that other games will not supply. DILUTES portability even where it serves
perception.

## 2. Chooser-bypass inventory (`_choose_action` path)

| Channel | Line | Gated? | What it does |
|---|---|---|---|
| **Exploration floor** | sim:1362-1364 | **NO** | epsilon = `0.025 + plasticity*0.055 + perturbation_rate*0.25` → **2.9-5.7% for agents, hard floor 2.5%**. Uniform-random over all 15 actions, skips both the controller's ranking AND the feasibility walk (handlers no-op infeasible attempts at small cost, per the comment at 1380-1384). This is what keeps a degenerate policy alive: a constant-output controller still eats, crafts, coordinates, and clones at epsilon rate. |
| Feasibility walk | sim:1371-1387 | YES (`action_search_depth`, config.py:71-82) | depth 0 = full ranked walk, world filters feasibility for free (the #2867 leak); k=1 forces the controller to rank an executable action first. |
| Drive injection | sim:1366-1370 | YES (`drive_injection_scale`, config.py:83-90) | adds `valence_spawn*(energy_ratio-0.62)*scale*(0.9..1.3)` (≤~0.37) to coordinate/clone_perturb outputs when adult and energy_ratio>0.62. **Verified: this is where spawn timing is conditioned on energy state harness-side.** The same state is in the observation (sim:1291), so 0.0 leaves it learnable. |
| Non-neural scripted branch | sim:1348-1356 | **NO** | controller-None individuals get a hardcoded policy incl. rng-driven clone_perturb at 2.5/3.5% (1349-1351). Applies to collectors/converters (the residue farm, load-bearing) but also to any "agent" whose `neural_budget` perturbs below 2 (individuals.py:435, optimization.py:111-112) — a scripted refuge lineage, though it can only rest+clone so it slowly runs out of energy. |
| **Action-argument targeting (systemic)** | — | **NO** | The controller picks one of 15 verbs; every *argument* is chosen by scripted state-reading heuristics: move destination (sim:1468-1477, place_memory+crowding scorer), craft target affordance (sim:1711-1748 — including a direct read of `challenge.expected_affordance()` at 1712-1716), craft components (1750-1775), use_tool affordance via `_situation_affordance_choice` (2028-2029, 1032-1084 — salient-problem detection + memory bias all harness-side), drain target = weakest local (2335), pickup material = random (1706), mark-read candidate (2944-2952). **This is the largest ungated perception leak: the perceiving is done by the harness; the NN only has to emit a verb.** |
| Combine-intent persistence | sim:348-351, 2398 | **NO** | one coordinate action opens a 6-19 tick window (`window = 6 + signal*8 + selectivity*5`, 2395) during which pairing resolves passively each tick regardless of subsequent controller outputs. Epsilon fires coordinate ~0.2-0.4%/tick → with the window and harness pairing (`_resolve_combine` 2729-2795) a blind policy reproduces via combine without ever ranking coordinate. **Ungated parallel to the drive-injection leak.** |
| Helper recruitment | sim:512-542, 586-593, 1601-1620 | **NO** | co-located individuals are pulled into `support` (income, skill gains, social feedback) and even relocated (`_move_expedition_helpers`) by harness code; their controllers' "choice" is at most a stale `last_action` membership test (524). |
| Place-memory steering | sim:3118-3133 → 1468-1475 | **NO** | harness writes a value map per place and the scripted move chooser consumes it; perception→action loop closed entirely outside the NN. |
| Rest replay | sim:1424-1425 | n/a | substrate channel into the controller, not a bypass of outputs. |
| Forward path | backends/cpu.py:11-12 | clean | no noise injected between controller outputs and ranking (controller-internal attention noise at controller.py:309-310 is part of the network). |

## 3. Champion / checkpoint categories

Selected in `_checkpoint_champions` (sim:3313-3363) at every
`checkpoint_every` tick and at final (sim:270, 423-424); buckets and quotas
in `checkpoints.py:22-35`. **None of these feed back into the world** — they
determine *what gets extracted*, which is the deliverable, so mis-aimed
metrics here corrupt the product directly.

| Category | Metric | Line | Perception-coupled? |
|---|---|---|---|
| overall_champion | `_checkpoint_score`: `child_count*6 + successful_tools*2 + cycle*0.75 + age/450 + energy_ratio*2 + complexity*0.5 + log-profile` (profile weights: causal_unlock 2.4, spawning 1.6, structure 1.5, tool_use 1.4, prediction_fit 1.3, ...) | sim:3245-3269, 3319-3321 | **DILUTE** — accumulation index. child_count and age dominate; both collectible blind (epsilon + drive/intent channels). `complexity*0.5` even rewards raw capacity. |
| spawn_champion | `(child_count, cycle, energy, age)` | sim:3323-3326 | **DILUTE** — pure accumulation; spawn timing historically harness-carried (TRANSFER_BARRIER #2867). |
| tool_champion | `(successful_tools, child_count, energy, age)` | sim:3328-3331 | **DILUTE** — `successful_tools` increments on any success (individuals.py:301-303) while affordance targeting is harness-side (§2); a blind verb-emitter accrues it. |
| causal_champion | `(causal_unlock, causal_step, successful_tools, energy)` | sim:3333-3345 | **DILUTE in current form** — sequences demand the right affordance (sim:2217-2225), but the harness leaks `expected_affordance` into craft targeting (1712-1716) and situation choice; the sequencing competence is substantially scripted. Would SERVE if targeting moved into the controller. |
| learner_champion | `(prediction_fit, -Σ|pred errors|, age, energy)` | sim:3347-3359 | **Closest to SERVING** — prediction heads are in the extracted NN and must fit real outcomes. But `prediction_fit` accrues +0.04 every tick that avg error <1.0 and anything happened (individuals.py:262-264) → longevity-confounded counter, not a rate. |
| line_founder | `(cycle, child_count, energy, age)` | sim:3361-3363 | **DILUTE** — lineage accumulation. |
| first_tool | first success per affordance | checkpoints.py:77-83; saved at sim:1887, 2024, 2105 | DEAD SURFACE for the axis (novelty bookkeeping; epsilon collects it). |
| notable_death | child≥3 or tools≥2 or causal/prediction/written thresholds | sim:3149-3168 | DEAD SURFACE for selection; archive recall only. |
| `member_score`/`line_score` | sim:3423-3450 | log/debrief only | DEAD SURFACE. |

## 4. The Optimizer's "quality score"

**There is none.** `Optimizer` (optimization.py:32-168) contains only
variation operators — clone/combine planning, costs, template inheritance —
triggered by in-world actions ("sealed_run_policy", optimization.py:166). It
never ranks individuals. The only explicit ranking in the codebase is
`_checkpoint_score` (§3), which runs at interval/death/final and affects
extraction only. In-world, an individual "ranks higher" exclusively by
staying above the energy/health deactivation lines and crossing spawn
thresholds — plus one social leak: `_partner_score` (sim:2797-2807) prefers
partners by health, energy ratio, mobility, manipulator, skill breadth, and
`child_count/5` — a mild rich-get-richer pairing term reading accumulation,
not perception (DILUTE, small).

## 5. Spawning economics

- **clone_perturb**: threshold `22 + single_parent_threshold*45 + complexity*7` ≈ 40-65 energy for agents (individuals.py:195-196; storage limit 24-144, individuals.py:133-134). Cost `threshold*(0.32+child_investment*0.28)` ≈ 13-30, reserve `max(threshold, cost*1.04)` (optimization.py:53-59). Child energy `cost*~0.42` ≈ 6-12 (optimization.py:63) — a parental subsidy that must carry the juvenile through 25 ticks to adulthood (individuals.py:192-193) at ~0.06-0.16/tick upkeep; comfortable margin, juveniles mostly deactivate from stress not energy.
- **combine**: coordinate gate at `combine_reserve*0.82` (sim:2391); reserve = `combine_threshold*(0.34+ci*0.10)` ≈ **only ~0.34-0.41× of the 50-100 threshold** (optimization.py:47-48), per-parent cost `threshold*(0.035+ci*0.050)` ≈ 2-8 (optimization.py:107-108), child energy `4 + 0.85*(cost_a+cost_b)` (optimization.py:88). Compatibility: param distance <0.50 (optimization.py:50-51); unreciprocated pairing still passes 35% (sim:2771). **Combine is drastically cheaper than clone — plus the passive intent window (§2) — making it the path of least perceptual resistance to offspring.**
- **Developmental grace** (`--neural-grace-ticks/-floor`, cli.py:47-58 → config.py:39-47 → individuals.py:186-189): neural upkeep ramps from 0.35× to 1× over the window. Default OFF. It blunts *capacity-cost* pressure, not perception pressure — load-bearing only for capacity-growth experiments (its stated purpose, config.py:39-45). Orthogonal to k=1.
- **Other subsidies**: seeding energy (sim:182-192); death recycling `+2.0 + 0.35*energy` residue (sim:3146); the scripted non-neural birth rate 2.5/3.5% (sim:1349-1351) that keeps the collector/converter food base stocked — **load-bearing for boot**, agents eat the residue economy; `_refresh_world` deliberately preserves resources to avoid a free buff (sim:118-145, correct for the axis).
- **Drive injection, verified**: acts at sim:1366-1370, inside `_choose_action_from_outputs`, after the epsilon roll and before ranking; conditions coordinate/clone_perturb boosts on `energy_ratio>0.62` and adulthood — state the controller never needs to read, though it is available at observation index 0 (sim:1291). This is the channel that carries spawn timing; gated, 0.0 removes it.
- **What blunts k=1**: drive injection (gated), the combine intent window + cheap combine reserve (ungated), epsilon floor (ungated), harness-side pairing/partner choice (ungated). **What is load-bearing for boot**: residue farm + death recycling + child energy subsidy + seeding energy. Grace subsidy: neither — separate experiment knob.

## 6. Terms whose removal would most sharpen the perception signal

1. **Harness-side action-argument targeting** — biggest ungated leak. `_situation_affordance_choice` (sim:1032-1084), `_craft_target_affordance`'s read of `challenge.expected_affordance()` (sim:1712-1716), move-destination scoring (sim:1468-1477), drain targeting (sim:2335). Until the controller has to perceive *what to act on*, k=1 only forces it to perceive *whether a verb is executable*.
2. **Combine-intent persistence window + cheap combine reserve** (sim:348-351, 2395-2398; optimization.py:47-48) — the ungated twin of drive injection for spawn timing; epsilon + window + harness pairing yields blind combine spawning.
3. **Exploration floor constant 0.025** (sim:1362) — keeps degenerate policies collecting achievements and spawn intents; make it param-only (selectable to ~0) or anneal it.
4. **Champion metrics' accumulation terms** — `child_count*6`, `age/450`, `cycle*0.75`, `energy_ratio*2`, `complexity*0.5` in `_checkpoint_score` (sim:3262-3267) and the spawn/line/tool champion key tuples (sim:3323-3361). These choose what gets extracted; today they choose accumulators. Re-aim at prediction-fit *rate* and held-out probe performance.
5. **`planning` amplifier** (sim:458-471) — fold the underlying signals into observation only; stop converting them into scripted success bonuses.
6. **Helper recruitment** (sim:512-542, 1601-1620) and **`_partner_score` accumulation terms** (sim:2804) — passive income and pairing preference that read state on behalf of controllers.
7. Near-dead weight safely cuttable: `valence_social` channel (coef ≤0.35, magnitudes ≤0.15), `tool` feedback's observation/prediction plumbing if prediction heads are not the focus, first_tool/notable_death buckets, `member_score`/`line_score` debrief scoring.

Already-gated knobs confirmed correct and sufficient for their stated
scopes: `action_search_depth` (config.py:71-82) and `drive_injection_scale`
(config.py:83-90) — but note items 1-3 above are ungated channels the k=1
campaign does **not** close.
