# Plan: recordings exported with derived (bipolar) channels

Status: **draft, to confirm** (2026-11-19). Anticipation: no study has such a recording yet. SPEC.md, *Planned modules →
Recordings exported with derived channels (bipolar montages)*, points here.

## 1. Context

A clinical recording with a low-density montage is often exported with its channels **already derived**: the
subtraction is done in the file and the channel name says it.
- PSG derivations: `C3-M2`, `C4-M1`, `O1-M2`, `EEG F3-A2`, eyes `E1-M2` / `E2-M1`, chin `Chin1-Chin2`.
- Clinical bipolar chains: `Fp1-C3`, `C3-O1`, `Fp2-C4`, `C4-O2`, `T3-C3`.

An EDF says nothing else about the reference: the name is the only information. The rule below is deduced from the name,
never from the format (a BrainVision or Curry file exported with derived channels follows the same rule).

## 2. What happens today (checked on 2026-11-19, on the code and on the name functions, no recording)

Works as is:
- Tools 1 and 2 type `C3-M2`, `EEG C3-A2`, `Fp1-C3` as EEG. Tool 2 finds `E1-M2`, `E2-M1`, `Chin1-Chin2` as context channels.
- Tool 2's re-reference method is *None (keep as-is)* by default, the right choice. Tool 7 (and tool 0's step 6) follows it.
- Tools 6, 7, 8, 8bis, 9 work channel by channel: a derivation is a channel like another. At 12 channels or fewer every
  high-density adaptation is inert, and an EDF has no positions (no topography).
- Tool 3bis leaves the EOG of an EDF as recorded.

Does not:
1. **Tool 3bis, contralateral mastoid (default)**: `autoscoring_lib.electrode_side` reads the last digit of the name, so
   `C3-M2` is taken as a right electrode (the 2 of M2), `C4-M1` as a left one.
   - No mastoid channel in the configuration (the usual export): no mastoid found, fallback to tool 2's reference (*None*):
     the signal is right, but the notes say *no contralateral mastoid … reference declared in tool 2 used instead*.
   - **A mastoid channel also kept in tool 2** (M1, M2, A1, A2): `C3-M2` becomes C3 − M2 − M1, **silently wrong**. Same
     when tool 2 renamed `C3-M2` to `C3`.
2. **Tool 3bis menus and SleepFM** match plain electrode names: GSSC's consensus ticks (`GSSC_TRAINED_EEG`), the GSSC / YASA
   defaults (`C3`, `C4`) and SleepFM's `AASM_EEG` miss `C3-M2`. SleepFM then takes the first 3 channels of the list instead
   of the AASM derivations present.
3. **Tool 3bis offers `E1-M2` as an EEG electrode** when tool 2 did not declare it as a context channel (`EOG_NAME_RE` only
   knows `E1` alone).
4. **Tool 2 accepts *Average* or *Custom*** on derived channels without a word: a mean of derivations, or a reference
   subtracted twice. Tool 7 then applies it.
5. Minor, **tool 0** (participant without tool 2's configuration): `detect_channel_types` pre-ticks `E1-M2` / `E2-M1` as
   EEG (the M2 of the name matches the EEG list, and the EOG name rule only looks for `eog`). Tool 1 already uses
   `EOG|LOC|ROC|\bE1\b|\bE2\b`.

Not answerable without a real file (out of this plan, see 7): how clinical systems name their channels, tool 7's µV
thresholds on short bipolar derivations (lower amplitude), the models' reliability on `Fp1-C3` type derivations.

## 3. The rule: a channel already derived (one copy, `recording_lib`)

`recording_lib.split_derivation(name)` → `(electrode, reference part)` or `None`:
- The name, without an `EEG ` prefix and spaces around the dash, is `A-B`, where A and B are both electrode names:
  `KNOWN_EEG_CHANNEL_RE` matched on the whole token (without its literal `EEG`), the mastoids `LEFT_MASTOIDS` /
  `RIGHT_MASTOIDS` (TP9 / TP10 included) and the eye names `E1`, `E2`, `LOC`, `ROC`.
- `C3-M2` → `('C3', 'M2')`, `EEG C3-A2` → `('C3', 'A2')`, `Fp1-C3` → `('Fp1', 'C3')`, `E1-M2` → `('E1', 'M2')`.
- `C3`, `M2`, `C3-REF`, `C3-Ref`, `C3-AVG`, `EOG G`, `Chin1-Chin2` → `None` (a plain or referential name: unchanged).

**Two channels can be subtracted only when they share the same reference part**: both plain (the same reference of the
recording, today's case), or both `-B` with the same B. (C3 − Cz) − (M2 − Cz) = C3 − M2 is right, (C3 − M2) − M1 is not.
- So an export that writes its common reference in the name (`C3-Cz`, `C4-Cz`, `M1-Cz`, `M2-Cz`, or everything on `A1`
  with an `A2-A1` channel) is re-referenced as today: (C3 − A1) − (A2 − A1) = C3 − A2.
- A channel `A-B` with no channel of the same reference part to subtract is a **finished derivation**: used as recorded.

`recording_lib.reference_part(name)` = B, or `''` for a plain name. `recording_lib.can_subtract(channel, reference)` = the
rule above, for one channel and the channels of its reference.

The rule reads the **original names** (tool 2's remap keys), so a derivation renamed in tool 2 (`C3-M2` → `C3`) is still
recognised.

## 4. Changes

### Phase 1: the rule (`recording_lib`)
- `split_derivation`, `reference_part`, `can_subtract` (section 3). Quick unit test on the names above.

### Phase 2: tool 3bis and tool 0's step 3 (`autoscoring_lib`, notebook menus)
- With the contralateral mastoid, an electrode `A-B` (`load_eeg_contralateral`, `read_signals`):
  - `B` is already the contralateral mastoid of `A` (`C3-M2`, `C3-A2`): it is the AASM derivation, read as recorded. Label
    `C3-M2` (M1 / M2 written as today), `eeg_reference` = `contralateral mastoid: C3-M2, C4-M1 (already derived in the
    recording)`.
  - A contralateral mastoid with the same reference part exists (`M2-Cz` for `C3-Cz`, `A2-A1` for `C3-A1`): subtracted.
  - Otherwise (ipsilateral mastoid `C3-M1`, electrode-electrode `Fp1-C3`): a finished derivation, used as recorded
    (*Q2*), never given tool 2's reference as a fallback, with a note in the log, the notes and the report: *Fp1-C3 is a
    bipolar derivation, not the contralateral-mastoid derivation the models learnt: used as recorded, the scoring may be
    less reliable*.
  - A plain name: today's code, fallback included.
- With *As declared in tool 2* (`load_eeg`, `read_signals`): a reference that `can_subtract` refuses is not applied to
  that channel (*Q1*), said in the notes.
- A configuration mixing derived and plain channels: each channel follows its own case.
- `electrode_side` and `contralateral_refs` work on the electrode part `A` of a derived name.
- Menus: the GSSC consensus ticks, the GSSC / YASA defaults and SleepFM's choice compare the electrode part `A`,
  preferring a contralateral-mastoid derivation (`C3-M2` before `C3-O1`). SleepFM gets `C3:M2` (a name it knows); a
  bipolar `C3-O1` takes a generic name as today (`SLEEPFM_FALLBACK_BAS`).
- `eeg_options` leaves out a name whose electrode part is an eye (`E1-M2`) (*Q4* for the bipolar EEG ones).
- Skip with other settings (*Cross-cutting → Skip*): nothing new is recorded (the reference description already is), no
  change to `settings_differences`.
- **Every referential study stays byte-identical** (no derived name → the former code path).

### Phase 3: tool 2, Section 5
- *Save re-reference*: *Average* or *Custom* on channels that `can_subtract` refuses (their reference parts differ:
  `C3-M2` with `M2`, an average of `C3-M2` and `C4-M1`, of `Fp1-C3` and `C3-O1`) gets a warning in the summary, like the
  online-reference one: `⚠️ config. 1: C3-M2, C4-M1 are already derived in the recording: this re-reference would subtract
  a reference again. Keep None.` A warning, not a block (*Q3*). An average of `X-Cz` channels gets none.
- An info line in the panel of such a configuration (left list): *n channels already derived (C3-M2, …): None keeps them
  as recorded*.

### Phase 4: tool 0's channel typing (`edf_io_lib.detect_channel_types`)
- EOG by name: `eog|LOC|ROC|\bE1\b|\bE2\b`, tool 1's rule. Only tool 0's pre-ticks change, for such names.

### Phase 5: documentation
- SPEC: *Cross-cutting* (a short paragraph *Channels already derived*), §2 (the warning), §3bis (the reference of a derived
  channel, the menus), §0 (typing), *Tests*. The TODO removed from *Planned modules*.
- CLAUDE.md: one line under *Channel typing* (a channel already derived is never re-referenced by tool 3bis, tool 2 warns),
  to confirm.

## 5. Tests (synthetic only, no real data)

- Quick (`tool3bis`, `tool0`, `tool2`):
  - `split_derivation` / `reference_part` / `can_subtract` on the names of section 3, the common reference in the name
    included.
  - `electrode_side`, `contralateral_refs`, `eeg_options`, menu defaults with derived names.
  - A small EDF written by `edfio` with `C3-M2`, `C4-M1`, `O1-M2`, `E1-M2`, `E2-M1`, `Chin1-Chin2` (+ a variant that also
    keeps `M1` / `M2`, + a `X-Cz` common-reference variant): the signals given to GSSC, YASA and SleepFM checked value by
    value (as recorded for the derived ones, subtracted for the common-reference variant), the notes and `eeg_reference`.
- Standard: tool 2 driven on that small study, the warning shown with *Average*, absent with *None*. Tool 3bis on it (the
  three models run, the report says *already derived*).
- Non-regression: the existing 3bis, tool 0 and tool 2 tests unchanged; `tests/refactor_check.py` on the synthetic chain
  (every file identical). `tests/score_edf_equivalence.py` (by hand, `autoscore` environment) only if Phase 2 touches the
  path its studies take (they are referential: it should not).

Level proposed when implementing: **standard** on `tool2`, `tool3bis`, `tool0` + refactor_check. To confirm then.

## 6. Questions (recommendation first)

- **Q1**: tool 3bis with *As declared in tool 2*, a reference that `can_subtract` refuses: **not applied, said in the
  notes** (recommended, the subtraction would be wrong whatever tool 2 says, and tool 2 now warns) / follow tool 2.
  Tool 7 keeps following tool 2 (its gate is tool 2's warning).
- **Q2**: an ipsilateral-mastoid or electrode-electrode derivation given to the models: **used as recorded with a note**
  (recommended) / left out of the menus.
- **Q3**: tool 2 with *Average* / *Custom* on derived channels: **a warning** (recommended, a study may have a reason) /
  Save refused.
- **Q4**: a bipolar EEG derivation (`Fp1-C3`) in the 3bis menus: **offered, after the contralateral-mastoid ones**
  (recommended) / not offered.

## 7. Later, with a real clinical file

- Check the rule on its channel names (names only, no signal), and on a BrainVision exported by Analyzer with derived
  channels (whether its `[Comment]` still names an online reference).
- Tool 7's thresholds (flat, amplitude) on short bipolar derivations.
- Whether the scoring of the models on bipolar chains is good enough to be offered by default (Q4).
