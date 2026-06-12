# Action subsystem map (env-axis review input, 2026-06-12)

Agent-produced map, lightly edited. Axis under judgment: *the environment's
only job is to be a selection pressure whose sole deliverable is a portable,
perception-coupled NN policy that can be extracted, dropped into other games,
and reverse-engineered.* Companion file: `SHAPING_MAP.md`. Synthesis:
`../ENV_AXIS_REVIEW.md`.

---

## 0. Canonical action list and the decision pipeline

The canonical list is **15 actions**, defined at `individuals.py:52-68`,
indexed by `ACTION_INDEX` at `individuals.py:70`. The controller output layer
is sized `len(ACTIONS)` (`individuals.py:398, 413`), so slot position is
checkpoint-load-bearing.

| idx | action | idx | action | idx | action |
|---|---|---|---|---|---|
| 0 | rest | 5 | pickup | 10 | signal |
| 1 | move | 6 | craft | 11 | mark |
| 2 | eat | 7 | build | 12 | coordinate |
| 3 | absorb_solar | 8 | use_tool | 13 | clone_perturb |
| 4 | forage | 9 | drain | 14 | observe |

Pipeline per tick (`simulation.py:303-351`):
1. Upkeep + terrain stress applied before choice (`1149-1159`, `1175-1281`) — can deactivate before acting.
2. Observation built (`_observe`, `1283-1334`); controller forward; exploration noise `1362-1364` (random action with p ≈ 0.025 + plasticity·0.055 + perturbation·0.25 — exploration **bypasses the gate entirely**, relying on handler no-ops).
3. **Drive injection** (`1366-1370`): if adult and energy ratio > 0.62, the harness adds `valence_spawn`-scaled boosts to output slots 12/13. This is a free state-read oracle, gated by `drive_injection_scale` (`config.py:83-90`).
4. **Feasibility walk** (`1371-1387`): legacy `depth<=0` walks the full ranking and silently executes the first feasible action (free feasibility oracle); `k>=1` checks only top-k, otherwise commits `ranked[0]` and increments `infeasible_commits` (`1386`). Committed-infeasible actions are adjudicated by the handlers, each of which no-ops at a small energy cost.
5. `coordinate` is special-cased out of `_resolve_action` (`342-345`); everything else dispatches at `1408-1452`.

**Non-neural (scripted) branch** (`_choose_action`, `1347-1356`): collector →
`absorb_solar` (+ `clone_perturb` p=0.025 above reserve); converter → `eat`
72% / `forage` 28% (+ `clone_perturb` p=0.035); fallback `rest`. So scripted
kinds touch only 4 actions; **11 of 15 actions are neural-agent-only**: rest,
move, pickup, craft, build, use_tool, drain, signal, mark, coordinate,
observe.

---

## 1. What the observation vector contains (gate-observability ground truth)

`_observe` (`simulation.py:1283-1334`), 72 dims = 42 base + 8 trace + 6
prediction-error + 8 event-memory + 8 signal-values:

**Present:** energy ratio (1291), health (1292), age/2000 (1293), 7 place
resources (1285/1294), sealed_essence/160 (1295), crowding (1296), local
neural fraction (1297), **inventory_count/inventory_limit (1298)**,
mobility/manipulator/resilience/sensor_range (1299-1302), 4 cognition params
(1303-1306), last_valence (1307), skill-breadth scalar (1308), season/climate
(1309-1310), 12 physics fields (1311-1322), 4 terrain fields (1323-1326),
recent trace (1327), prediction errors (1328), event memory (1329), learned
signal-token values (1330).

**Absent:** artifact count, artifact capabilities, artifact durability;
inventory composition or absolute count; place material stocks; place
obstacle levels (only proxies: fluid/temperature/elevation); **causal
challenge existence, sequence, or progress**; structure presence (only
indirectly: `_apply_structures` writes enclose/permeable/shelter back into
`place.physics` at `world.py:674-676`, which lands in obs dims 1314-1316);
helper states / collective material count; lesson memory; signal/mark
*content* (only token identity routed through the learned `signal_values`
channel via `_observed_tokens`, `1336-1345`).

---

## 2. Per-action map

Legend for verdicts: **S** = SERVES axis, **D** = DILUTES, **X** = DEAD SURFACE.

| idx | action | gate (line) | gate observable? | handler micro-targeting | direct energy payoff | kinds | weight (lines) | verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | rest | none | — | none | −0.004 | agent (+fallback) | ~12 | S |
| 1 | move | mobility ≥ 0.05 (1400) | YES (1299) | handler picks destination | none (relocation) | agent | ~323 | S core / X telemetry |
| 2 | eat | none | — | none | essence+residue gain | agent, converter | ~30 | **S** |
| 3 | absorb_solar | none | — | none | solar gain | agent, collector | ~9 | **S** |
| 4 | forage | none | — | random find | none (seeds place) | agent, converter | ~11 | D |
| 5 | pickup | inv_count ≥ inv_limit (1392) | **YES — exactly obs dim 14 (1298)** | random material (1706) | −0.035, +1 material | agent | ~13 | S |
| 6 | craft | inv_count<2 OR artifacts ≥ limit (1394) | PARTIAL / **NO** | handler picks target + components | −0.12..0.28, artifact | agent | ~190 (+375 energy.py) | D |
| 7 | build | manip<0.18 (1396); inv<3 AND collective<3 (1398) | YES / **NO** | handler picks extend-vs-new, target structure | −0.16..0.44, structure | agent | ~175 (+~280) | D |
| 8 | use_tool | inv==0 AND no artifacts (1402) | PARTIAL | handler picks affordance + effect | up to ~15+/tick | agent | ~483 | S payoff / D wiring |
| 9 | drain | none | — | handler picks weakest target (2335) | 30-75% of target energy on deactivation | agent | ~72 | S |
| 10 | signal | none | — | token = argmax(outputs[0:8]) | −0.025.. | agent | ~36 | S (weak) |
| 11 | mark | manipulator ≥ 0.12 (1404) | YES (1300) | handler picks lesson, portable-vs-place | none (cost only) | agent | ~525 (with read side) | **X** |
| 12 | coordinate | adult (1390) | YES (1292-1293) | handler picks partner (2763) | child (spawning) | agent | ~103 | S |
| 13 | clone_perturb | adult (1390) | YES | optimizer plans child | child (spawning) | all kinds | ~90 | S |
| 14 | observe | none | — | random demo; best mark auto-picked | none (skill gains) | agent | ~18 (+204 mark-read) | D |

### Per-action detail

**rest (0)** — `1415-1426`. Cost 0.004; if controller has episodic memory,
one replay step (`1424-1426`). Always feasible: it is the gate-walk floor in
legacy mode (`1379`). Verdict S: the do-nothing arm that makes action choice
meaningful; replay is controller-internal substrate, zero world surface.

**move (1)** — `_move` `1454-1599`. Gate reads `params.mobility` (`1400`),
observable (`1299`). Handler: destination chosen *by the handler* —
memory-greedy over neighbors for controllers with place memory
(`1468-1475`), else `rng.choice` (`1477`). The controller decides only
*whether* to move, never *where*. Cost: base `1467`; success cost
`1531-1539` (barrier, distance, uphill, pressure, danger, relocation shock ×
0.58, load); failure cost+damage `1572-1576`; deaths
`relocation_shock`/`movement_hazard` (`1570, 1599`). Support machinery:
`_movement_motivation` (`683-706`), `_relocation_shock` (`708-750`),
`_record_movement` (`752-816`), `_movement_summary` (`818-838`),
`_move_expedition_helpers` (`1601-1620`), 4 Counter fields (`96-99`).
Verdict: the leave/stay decision is genuinely perception-coupled (hazard,
crowding, resources all observable); **but** ~90 lines of motive
attribution/telemetry (`752-838`) are write-only (X), and motivation/shock
modeling (~70 lines) exists to price a choice the controller doesn't make.

**eat (2)** — `1663-1673`. No gate. Gain `1670` from observable resources
(obs dims 4-5). Triggers patch recovery (`1672-1673` → `world.py:863-879`),
which makes feeding *timing/location* a learnable, perception-dependent
competence (the pre-registered jitter control, `config.py:62-70`). Verdict
**S** — the cleanest perception-or-die loop in the file.

**absorb_solar (3)** — `1675-1683`. No gate. Gain from observable solar;
thermal stress health cost (`1678`) from observable thermal/temperature.
Verdict **S**.

**forage (4)** — `1685-1695`. No gate. Costs energy; with probability keyed
to `sensor_range` *adds resources and materials to the place* — it never
feeds the actor directly. The payoff is a public good harvested by whoever
eats/picks up next. Success chance is a params lottery, not a perception
read. Verdict **D**: reward variance plus a world subsidy; the only axis
value is seeding materials for the craft chain.

**pickup (5)** — `1697-1709`. Gate: inventory full (`1392`) — **this gate
input is literally observation dim 14** (`1298` is exactly
`inventory_count()/inventory_limit()`; gate trips at 1.0). Hidden handler
gates: `manipulator < 0.08` (`1698`, observable) and *place has no
materials* (`1702-1704`, NOT observable — place materials never enter the
obs). Picks a `rng.choice` material; the controller cannot select what to
pick up. Verdict S with a caveat: the infeasible-commit tax on pickup is a
pure ranking failure on an observable input — exactly the demand k=1 is
supposed to create — but the empty-place no-op (−0.015) is unresolvable by
perception.

**craft (6)** — `_craft` `1777-1887`. Gate (`1394`): `inventory_count < 2`
OR `len(artifacts) >= artifact_limit()`.
- *inventory half:* PARTIAL — obs has only the ratio; the limit varies with
  `developmental_complexity` and carried `carry` capability
  (`individuals.py:143-146`), neither observable, so "count ≥ 2" is not
  recoverable except at ratio==0.
- *artifact half:* **NOT OBSERVABLE** — no artifact features exist in
  `_observe`.
Handler micro-targeting is total: target affordance scored against place
state by `_craft_target_affordance` (`1711-1748` — reads causal challenge,
exposure, obstacles, resources the controller mostly can't see), components
chosen by `_select_craft_components`/`_component_craft_score`
(`1750-1775`). Cost `1822`; failure breaks components (`2308-2322`); success
builds an `Artifact` (`energy.py:438-470`), increments `artifacts_created`,
`tool_make`, feeds `save_first_tool` (`1887`). Verdict **D**: the
artifact→survival chain is real (see §3.1), but the cognition lives in ~75
lines of handler scripting, the gate is half-blind, and success chance is
mostly params+skill (`1813-1821`).

**build (7)** — `_build_structure` `1889-2024`. Gates (`1396-1398`):
`manipulator ≥ 0.18` (observable) and `inventory ≥ 3 OR
collective_material_count ≥ 3`. `_collective_material_count` (`1622-1626`)
sums up to 4 helpers' inventories, where helper candidacy depends on their
signals/last actions/combine intents (`512-542`) — **NOT OBSERVABLE** at any
fidelity. Handler: random component draws (`1905-1912`), extends the largest
existing structure 62% of the time (`1962-1968`), else builds new. Output is
a *place-level public good*: structures shelter everyone present
(`1185-1205`), passively generate resources (`world.py:614-636`), resist
drains (`560-570`) — the builder pays energy+materials, the place collects.
Verdict **D**: free-rider economics dilute individual selection; gate half
is perception-proof; plus a large dead annex (decay model, §3.2).

**use_tool (8)** — `_use_tool` `2026-2132`. Gate (`1402`): infeasible iff
inventory empty AND no artifacts; inventory==0 is observable (ratio==0 at
dim 14), artifact possession is not — so the gate is conservatively
inferable but not decidable. Even when feasible, `score < 0.08` no-ops at
−0.05 (`2031-2032`), and score depends on inventory composition + artifact
capabilities (`energy.py:660-673`), both unobservable. Handler
micro-targeting is the deepest in the codebase: `best_affordance` picks from
inventory/artifacts, then `_situation_affordance_choice` (`1032-1084`)
overrides toward `_salient_problem` (`840-906`) using place state plus
`_memory_affordance_bias` over lesson memory (`1012-1030`) — ~210 lines of
scripted situation-cognition the controller never expresses. Payoffs
(`_tool_effect`, `2152-2197`) are the largest in the game: cleave releases
up to `2+9·competence` sealed essence (`2154-2157`); ferry taps
electrical+dense_node (`2175-2185`); plus causal release up to
`3+13·competence+…` (`2248`). The driving state (sealed_essence, resources)
**is observable** (obs dims 4-11). Verdict: **S in payoff topology / D in
wiring** — it creates the right gradient (perceive sealed_essence → rank
use_tool → big energy), but the handler answers the hard perception
questions (which affordance, which sequence step) itself.

**drain (9)** — `2324-2373`. No gate. Handler auto-picks the weakest
co-located target (`2335`). Cost `0.10+draw·0.08`; payoff only when the
target deactivates (`2370-2373`, 30-75% of target energy); risk: feedback
load up to −0.32 health, deactivation by `overload` (`2364-2369`). Crowding
is observable (dim 12); target weakness is not. Verdict **S** (moderate):
creates a real co-located contest with observable preconditions and
deactivation feedback both ways; also the only action that turns other
individuals into energy, closing the loop on agent density.

**signal (10)** — `2375-2383`. No gate. Token = argmax over the **first 8
action-output slots** (`individuals.py:207-210`) — a quirky reuse of output
slots 0-7 as a vocabulary. Cost-only for the emitter; effects: signals feed
`_observed_tokens` → `learn_signal_value` → the 8-dim `signal_values` obs
block (`1330`), and raise helper candidacy (`521-536`). Verdict **S
weakly**: this is the *only* inter-individual channel that actually reaches
the observation vector, so it is the one social feature aligned with
perception-coupling; payoff to the emitter is nearly nil though.

**mark (11)** — write: `_mark` `2404-2455` + `_inscribe_portable_mark`
`2473-2515` + intent/lesson/encode chain `2517-2663` + portable aging
`1161-1173`; read: `_readable_mark_candidates` `2892-2936`,
`_read_mark_trace` `2938-3062`, `_apply_mark_author_feedback` `3064-3097`;
world: `Mark` `world.py:42-65`, `create_mark` `881-899`, erosion `843-854`.
Gate manipulator≥0.12 (observable). Energy: writer pays up to ~0.4 (`2416`);
reader pays ~0.01-0.02 and gains *skill increments* with `gain` typically
≤0.04 (`2989-2996`) — never energy. `written_learning`/
`knowledge_transmitted` feed only checkpoint scoring (`3257-3258`) and
notable-death gating (`3155-3156`). Eight dedicated Counters (`86-92`). The
clarity-tiered lesson encoding (`2630-2663`) produces dicts no controller
can ever perceive — lessons live outside the obs vector and are consumed
only by other handler-side scripts (`1012-1030`, `2938-3062`). Verdict
**X**: ~525 lines of write-mostly bookkeeping whose survival feedback rounds
to zero and whose information content is invisible to the deliverable
policy.

**coordinate (12)** — `_coordinate_combine` `2385-2402` + `_resolve_combine`
`2729-2795` + `_partner_score` `2797-2807` + `_combine` `2812-2854`. Gate:
adult (`1390`), observable (age dim 3, health dim 2 — though age 25 maps to
0.0125 on the /2000 scaling, a weak signal). Handler-side energy checks
(`2391, 2748-2757`) read thresholds built from unobservable params
(`individuals.py:195-205`); energy ratio itself is observable. Partner
choice is scripted (`2763`). Verdict **S**: spawning is the selection
currency itself; the demand "rank coordinate only when adult+rich" is
perception-meetable — *provided* drive injection (`1366-1370`) is off,
otherwise the harness answers it for free.

**clone_perturb (13)** — `2665-2727` + `optimization.py:53-76`. Gate adult
(observable); handler checks pool cap, local place capacity (`2672`,
crowding observable at dim 12), and the optimizer's reserve
(`optimization.py:57-59`, threshold ≥ ~22 energy, partly unobservable
params). Parent pays a large real cost (`optimization.py:56`). Verdict **S**
— same logic as coordinate; also the only spawn path scripted kinds use.

**observe (14)** — `_observe_others` `3099-3116`. No gate. Consumes
same-tick `demonstrations` (cleared at `290`, filled by build/use_tool at
`1944/1989/2058/2121`) — whether a demo exists is unknowable at choice time
— plus the mark-read chain. Gains: skill increments of order 0.004-0.012
(`3110-3111`). Verdict **D**: an action slot whose payoff is
invisible-at-choice-time and tiny; it exists to give the
marks/demonstration subsystems a consumer.

---

## 3. Support subsystems

### 3.1 Crafting / recipes / artifacts — ~565 lines
State: `Individual.inventory`, `Individual.artifacts`
(`individuals.py:93-94`), `MATERIALS` (7 materials × ~20 properties,
`energy.py:117-283`), `Artifact` (`energy.py:43-75`), counters
`artifacts_created/broken` (`100-101`). Fed by: pickup, forage (materials),
craft; consumed by use_tool, mark (record artifacts), and passively by
carry/protect/insulate/float/anchor/traverse capability reads in terrain
stress (`1189-1205`), movement (`1487-1490`), physics transport
(`1100-1102`), drain resistance (`2337-2339`). Wear: `_wear_artifacts`
(`2295-2306`); break returns 0.1 essence. **Survival feedback: real and
multi-path** — artifacts measurably reduce deactivation rates and raise tool
gains. **Axis verdict: D** — the feedback is real but the entire artifact
state is *invisible to the policy* (zero obs dims), so every
artifact-conditional behavior must be carried by handler scripts or by
accident; a perception-coupled policy cannot even know it is holding a tool.

### 3.2 Building / structures — ~475 lines
State: `Place.structures`, `Structure` (`energy.py:77-114`), capability
derivation (`477-516`), decay model (`structure_decay_channels`,
`energy.py:583-657`, 8 channels), world-side per-tick processing
`_apply_structures` (`world.py:582-676`), counters
`structures_built/extended`. Feedback: passive resource generation
(`world.py:614-636`), shelter/interiority written into physics (→
observable, dims 1314-1316), stress reduction (`1185-1205`), movement
support (`1481-1490`). **Axis verdict: D, with the decay model X** — the
public-good economics blunt individual selection; the 8-channel decay
simulation modulates durability that no one can perceive, producing zero
behavioral demand. The one axis-positive part: structures alter *observable*
physics fields, so their effects (not their existence) are perceivable.

### 3.3 Artifact lifecycle (creation/breaking/decay)
Counted inside 3.1/3.2; standalone pieces are `_wear_artifacts`
(`2295-2306`), portable-inscription aging (`1161-1173`), break/decay
counters. Durability is unobservable end to end. X-leaning.

### 3.4 Marks (creation/reading/lessons/portable) — ~525 lines + 8 counters
Covered under action 11. State: `Place.marks`, `Artifact.inscriptions`,
`Individual.lesson_memory` (cap 5, `individuals.py:305-308`), counters
`86-92`. Fed by mark; read by observe; erodes in `world.py:843-854`.
Feedback into survival: skill drips ≤~0.04/read; author feedback ≤0.020
inscribe-skill and requires co-location (`3070-3072`).
`written_learning`/`knowledge_transmitted` feed checkpoint score and
notability only. **Verdict: X — the single largest dead surface in the
file.** A parallel cultural-transmission channel built entirely out of
handler-side dict plumbing, invisible to the obs vector, with no meaningful
energy/health/spawn coupling.

### 3.5 Signals — ~60 lines
State: `Place.signals`, `Individual.signal_values` (8 dims, in obs),
`coordination_token`. Fed by signal, coordinate, (tokens also from marks).
Feedback: helper candidacy (`521-536`), combine pairing context, signal
advection (`world.py:827-841`). **Verdict: S (weak)** — uniquely, its output
lands in the observation vector and is valence-trained (`413-414`), making
it the one social subsystem a portable policy could actually internalize.
Cheap to keep.

### 3.6 Tools / tool-skill (lash/hoist/kindle/cleave/encase/ferry/winnow/shear) — ~535 lines
State: `tool_skill` dict (~20 keys/individual, `individuals.py:95`),
`tool_use_counts`, `successful_tools`, `SKILL_TRANSFER` (`44-60`),
`_increase_skill` (`495-507`), `_skill_breadth` (`473-493`, summarized into
one obs dim at 1308). Fed by nearly every handler (≈40 call sites).
Feedback: skill raises success chances in craft/build/use_tool and feeds
`planning` (`_interaction_control`, `458-471`) which multiplies efficiency
everywhere. **Verdict: D** — this is a second, *non-neural* adaptation
channel: a lifetime-accumulated scalar table that improves outcomes without
any perception or controller involvement, and it is not heritable, so it
doesn't even compound across the selection loop. It competes with the
controller for credit: a "rank use_tool always" policy gets better over a
lifetime via skill, masking perception differences. The transfer matrix
(`44-60`) is pure handler intelligence.

### 3.7 Causal unlocks / causal steps — ~190 lines
State: `Place.causal_challenge` (`world.py:68-95`), generation
`world.py:467-522` (sequences of 2-5 affordances + prep steps keyed to place
physics), advancement `_advance_causal_challenge` (`2199-2293`), counters,
a checkpoint bucket (`checkpoints.py:30`). Fed by use_tool only. Feedback:
**direct, large energy release** (`2248-2250`) into observable place
resources. **Verdict: in principle the flagship S, in practice D.** The
challenge's existence, required affordance, sequence, and progress appear
*nowhere* in `_observe`; the expected step is read by
`_craft_target_affordance` (`1712-1716`) and `_salient_problem` (`842-854`)
— handler scripts. Multi-world refresh (`118-178`) explicitly exists to
force re-learning of these rules, but the policy has no sensory channel to
learn them through; only the handler's lesson-memory machinery does.

### 3.8 Collaboration events — ~180 lines
`_active_helper_candidates` (`512-542`), `_collective_support` (`544-554`),
`_apply_collaboration_effects` (`579-608`), `_agent_resistance_context`
(`556-577`), expedition helpers (`1601-1620`), collective build components
(`1628-1661`), counter `collaboration_events`. Fed by
move/build/use_tool/drain/terrain-stress. Feedback: support raises success
and reduces damage; helpers pay tiny energy (`590`). **Verdict: D** —
support is computed from helper states the actor cannot perceive, so it
injects success-probability variance that perception cannot explain; helper
selection and contribution are fully scripted.

### 3.9 Spawning machinery (coordinate/clone_perturb/combine) — ~330 lines incl. optimization.py
The selection operator itself. **S by definition** — but note both
free-state channels documented at `config.py:71-90` route around it: the
drive injection conditions spawn timing on unperceived state, and the legacy
feasibility walk hid the adult gate. With `drive_injection_scale=0` and
`k=1`, spawning timing becomes a genuine perception demand (energy ratio dim
0, age dim 2, health dim 1 are all in the obs).

### 3.10 Checkpointing categories — ~215 lines
`checkpoints.py` (buckets: first_tool, interval/spawn/tool/causal/learner
champions, line_founder, notable_death, `22-35`), `_checkpoint_score`
(`3245-3269`), `_checkpoint_champions` (`3313-3363`), `save_first_tool` call
sites in craft/build/use_tool (`1887, 2024, 2105`). No selection feedback at
all — this is the **extraction pipeline**, i.e. the deliverable side of the
axis, not the pressure side. Keep, but note `_checkpoint_score` weights
mark/structure/social categories (`3253-3258`) that the verdicts above rate
D/X — champion selection currently rewards behaviors the axis doesn't
value.

### 3.11 Telemetry/observer surface (cross-cutting) — ~400+ lines
Every handler carries an `observer.observe` block (e.g. `1871-1886`,
`2001-2017`, `2069-2090`, `2436-2454`, `2499-2515`, `3038-3061`,
`3080-3097`), plus movement recording (`752-838`) and the success_profile
system (`individuals.py:33-46, 294-299`) whose only consumers are checkpoint
scoring and notability gates. Write-only with respect to selection. **X** as
world-mechanics, though some is legitimately the observability deliverable.

---

## 4. Verification of the k=1 / infeasible-tax claim

**Mechanics: confirmed.** `simulation.py:1371-1387` commits `ranked[0]` when
the top-k window has no feasible action; every relevant handler no-ops at
cost: pickup −0.025/−0.015 (`1699, 1704`), craft −0.025/−0.020 (`1779, 1783,
1790`), build −0.035/−0.025 (`1891, 1898, 1902, 1918`), use_tool −0.05
(`2031-2032`), mark −0.02 (`2406`), coordinate −0.015/−0.020 (`2389, 2393`),
clone_perturb −0.02/−0.015 + plan penalty (`2669, 2674, 2679`).

**Concentration: confirmed, slightly understated.** From the three runs
under `kaggle/results/`:
- `mg-percept-v-s341`: use_tool/craft/build/pickup = 4,523 / 5,241 = **86.3%**
- `mg-percept-v-s44`: 151,949 / 177,420 = **85.6%**
- `mg-percept-v-s45b`: 56,777 / 73,166 = **77.6%**
- pooled: 213,249 / 255,827 = **83.4%**. use_tool alone is 43-59% of the tax
  in every run.

**Observability of the gate inputs: confirmed with one correction.**
- artifact count / artifact_limit (craft, `1394`; use_tool half, `1402`):
  **NOT observable** — no artifact feature exists in `_observe`. Confirmed.
- collective material count (build, `1398`): **NOT observable** — depends on
  unperceivable helper states (`512-542`, `1622-1626`). Confirmed.
- **Correction — pickup:** its gate input *is* observable, exactly: obs dim
  14 is `inventory_count()/inventory_limit()` (`1298`) and the gate (`1392`)
  trips at 1.0. Pickup's share of the tax (5-18% per run) is a *ranking*
  failure on a visible input — which is precisely the failure k=1 is
  designed to punish, so pickup tax is signal, not noise. Likewise the
  inventory halves of craft (`<2`) and use_tool (`==0`): ratio==0 is
  decidable, but "count==1 vs 2" is not, because `inventory_limit` varies
  with unobserved `developmental_complexity` and `carry` capability
  (`individuals.py:143-146`).

**Net:** of the four taxed actions, one gate is fully perceivable (pickup),
two are half-perceivable (craft, use_tool), one is unperceivable in its
rescue clause (build). The unobservable halves all trace to the same root
cause: **artifact state and other-individual state have zero observation
dims.**

---

## 5. Verdict summary

| unit | verdict | one-line evidence |
|---|---|---|
| eat / absorb_solar / patch recovery | **SERVES** | direct energy from observable resources; jittered recovery is a pre-registered perception probe (`1663-1683`, `world.py:863-879`) |
| use_tool energy core (release of sealed reserves keyed to observable resources) | **SERVES** | largest payoffs keyed to obs dims 4-11 (`2152-2197`) |
| spawning (coordinate/clone_perturb) with k=1 + drive_injection=0 | **SERVES** | adult+energy gates fully observable (dims 0-3); injection at `1366-1370` is the leak |
| move (decision core) | SERVES | hazards/crowding observable; destination scripting caps the demand (`1468-1477`) |
| drain, signal, rest | SERVES (minor) | contest with observable crowding; only obs-reaching social channel; no-op floor |
| pickup | SERVES under k=1 | gate == obs dim 14 (`1298` vs `1392`) |
| craft / artifact system | **DILUTES** | real survival chain, but zero artifact obs dims + ~75 lines of handler target cognition (`1711-1775`) |
| build / structures | **DILUTES** | public-good payoff + unobservable collective gate (`1398`, `1622-1626`) |
| use_tool wrapping (situation choice, lesson bias) | DILUTES | ~210 lines of scripted cognition answer the perception question for the policy (`840-1084`) |
| causal challenges | DILUTES (convertible to SERVES) | direct energy, but challenge state has zero obs dims; consumed only by handler scripts (`842-854`, `1712-1716`) |
| tool_skill / SKILL_TRANSFER | DILUTES | non-neural, non-heritable adaptation channel competing with the controller (`44-60`, `495-507`) |
| collaboration | DILUTES | support from unperceivable helper state injects unexplainable success variance (`512-554`) |
| forage, observe | DILUTES | params-lottery place seeding; invisible-at-choice-time payoff (`1685-1695`, `3099-3116`) |
| marks (entire write+read+portable+lesson-encode chain) | **DEAD SURFACE** | ~525 lines, 8 counters; payoff = skill drips ≤0.04; content invisible to obs; feeds only checkpoint/notability scores (`2404-2663`, `2892-3097`) |
| structure decay model (8 channels) | DEAD SURFACE | 75+ lines modulating unperceivable durability (`energy.py:583-657`) |
| movement telemetry, observer blocks, success_profile | DEAD SURFACE | write-only wrt selection (`752-838`, `individuals.py:33-46`); some is the legitimate extraction/observability deliverable |
| checkpointing | orthogonal (deliverable side) | but `_checkpoint_score` weights D/X behaviors (`3253-3258`) |
