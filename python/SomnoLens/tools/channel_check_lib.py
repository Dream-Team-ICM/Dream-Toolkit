"""Checks of the channels a recording gives to the tools: they warn, they never decide or correct anything. Every format
(read through recording_lib), no torch, so tools 0, 3bis, 6 and 7 can import it.

- channel_quality_warnings: an EEG electrode, a mastoid or an EOG that is noisy or dead during the recording. A bad
  mastoid is the worst case: C3-M2 = C3 - M2, so all the noise of M2 enters every derivation that uses it, with no visible
  sign of where it comes from. The impedance measured before the recording is not enough: a channel can be bad with a
  good impedance.
- context_amplitude_warnings: an ECG or a chin EMG whose amplitude is far from the physiological range. It usually means
  an input with its own gain (an auxiliary input behind an adapter) that the file does not declare. YASA's EMG features
  are absolute amplitudes, so such a gain changes its scoring. The factor given is an order of magnitude only.
- context_gain_advice: tool 2's line on that amplitude, from the first recordings of a channel configuration, which
  proposes the factor its 'Divide by' field corrects (edf_io_lib.context_gain).

A context channel is judged after the correction set in tool 2 (divided by its factor), and the warning says so.
The thresholds come from physiology and were checked on EDF, Curry and BrainVision recordings (SPEC, Channel checks).
"""
import math

import numpy as np
from scipy import signal

import recording_lib as RL
from edf_io_lib import context_divisors, context_gain, drop_suffix_duplicates

EPOCH_S = 30
N_EPOCHS = 60             # epochs sampled over the recording for the EEG and EOG check
N_EPOCHS_CONTEXT = 30     # for the ECG and EMG check
BAND_HZ = (0.3, 35.0)     # what the models see of the EEG and EOG
NOISY_RATIO = 3.0         # EEG: standard deviation over 3 times the median EEG channel of the same epoch...
DEAD_RATIO = 0.1          # ...or under 0.1 times it (the electrodes next to the online reference go down to about 0.26)
SHARE = 0.2               # ...on more than 20 % of the sampled epochs
MIN_POOL = 3              # a median EEG channel needs at least 3 channels
EOG_MAX_UV = 100.0        # EOG: median standard deviation over the epochs
FLAT_UV = 1.0             # any channel: standard deviation under 1 uV is flat (no EEG, EOG or EMG is that small)
ECG_RANGE_MV = (0.1, 5.0)       # R wave, peak to peak, at the skin
ECG_TYPICAL_MV = 1.0
HEART_RATE_RANGE = (30, 120)    # beats per minute (median over the sampled epochs)
EMG_MAX_UV = 100.0              # chin EMG, RMS 10-100 Hz without the mains (median over the sampled epochs)
EMG_TYPICAL_UV = 5.0
MAINS_SHARE = 0.5               # the mains (50 or 60 Hz) holding more than half of the power


# ---------------------------------------------------------------------------------------------------------
# Signal measures
# ---------------------------------------------------------------------------------------------------------
def sampled_epochs(n_times, sfreq, n=N_EPOCHS):
    """Indexes of n 30-s epochs spread evenly over the recording (all of them in a shorter one)."""
    n_epochs = int(n_times // (EPOCH_S * sfreq))
    if n_epochs < 1:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, n_epochs - 1, min(n, n_epochs)).astype(int))


def epoch_std_uV(raw, picks, epochs):
    """(n epochs, n channels) standard deviation in uV of the channels `picks` of `raw` (lazy or loaded), each 30-s epoch
    filtered 0.3-35 Hz on its own."""
    sfreq = raw.info['sfreq']
    sos = signal.butter(4, [BAND_HZ[0], min(BAND_HZ[1], sfreq / 2 - 1)], btype='band', fs=sfreq, output='sos')
    n = int(round(EPOCH_S * sfreq))
    stds = np.zeros((len(epochs), len(picks)))
    for k, epoch in enumerate(epochs):
        data = raw.get_data(picks=picks, start=int(epoch) * n, stop=int(epoch) * n + n) * 1e6
        stds[k] = signal.sosfiltfilt(sos, data, axis=1).std(axis=1)
    return stds


def compared_with_median(stds, pool):
    """(noisy share, dead share) per column of `stds`: the share of the epochs where the channel is over NOISY_RATIO, or
    under DEAD_RATIO, times the median of the `pool` columns of the same epoch. None when the pool is too small."""
    if len(pool) < MIN_POOL:
        return None
    median = np.median(stds[:, pool], axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = stds / median
    return (ratio > NOISY_RATIO).mean(axis=0), (ratio < DEAD_RATIO).mean(axis=0)


def ecg_amplitude_uV(x_uV, sfreq):
    """(median R wave peak to peak in uV, heart rate per minute) of one ECG epoch, or (nan, nan) when fewer than 5 beats
    are found. Beats are found on the 5-30 Hz band (the QRS), measured on the 1-40 Hz band."""
    detect = signal.sosfiltfilt(signal.butter(4, [5, min(30, sfreq / 2 - 1)], btype='band', fs=sfreq, output='sos'), x_uV)
    shape = signal.sosfiltfilt(signal.butter(4, [1, min(40, sfreq / 2 - 1)], btype='band', fs=sfreq, output='sos'), x_uV)
    peaks, _ = signal.find_peaks(np.abs(detect), height=0.5 * np.percentile(np.abs(detect), 99.5),
                                 distance=int(0.35 * sfreq))
    if len(peaks) < 5:
        return np.nan, np.nan
    half = int(0.06 * sfreq)
    p2p = [np.ptp(shape[max(p - half, 0):p + half]) for p in peaks]
    return float(np.median(p2p)), float(60 * sfreq / np.median(np.diff(peaks)))


def emg_rms_uV(x_uV, sfreq, mains=50.0):
    """RMS in uV of a chin EMG epoch on 10-100 Hz, the mains and its first harmonic notched out."""
    y = signal.sosfiltfilt(signal.butter(4, [10, min(100, sfreq / 2 - 1)], btype='band', fs=sfreq, output='sos'), x_uV)
    for f0 in (mains, 2 * mains):
        if f0 < sfreq / 2 - 2:
            b, a = signal.iirnotch(f0, 15, fs=sfreq)
            y = signal.filtfilt(b, a, y)
    return float(np.sqrt(np.mean(y ** 2)))


def mains_share(x_uV, sfreq):
    """(share of the 1-100 Hz power within 1 Hz of the mains, mains frequency): 50 or 60 Hz, whichever holds more."""
    freqs, power = signal.welch(x_uV, fs=sfreq, nperseg=int(min(len(x_uV), 4 * sfreq)))
    band = (freqs >= 1) & (freqs <= min(100, sfreq / 2))
    total = power[band].sum()
    if total <= 0:
        return 0.0, 50
    shares = {f0: power[band & (np.abs(freqs - f0) <= 1)].sum() / total for f0 in (50, 60) if f0 < sfreq / 2}
    if not shares:
        return 0.0, 50
    f0 = max(shares, key=shares.get)
    return float(shares[f0]), f0


def decade(value):
    """The power of ten nearest to `value` on a log scale: 72 -> 100, 0.03 -> 0.01 (an order of magnitude)."""
    return 10.0 ** round(math.log10(value))


def divided_text(divide_by):
    """' after the correction set in tool 2 (divided by 100)', or '' without a correction."""
    return f' after the correction set in tool 2 (divided by {divide_by:g})' if divide_by != 1 else ''


# ---------------------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------------------
def _read(path, originals):
    """The channels `originals` (names in the file) opened lazily, without the online reference (a flat channel in the
    data as recorded), and {original name: name in the Raw} for those found (MNE renames duplicated names with -0)."""
    raw = RL.read_raw(path, include=list(dict.fromkeys(originals)), add_online_reference=False)
    raw, _ = drop_suffix_duplicates(raw)
    names = set(raw.ch_names)
    found = {}
    for orig in originals:
        if orig in names:
            found[orig] = orig
        elif f'{orig}-0' in names:
            found[orig] = f'{orig}-0'
    return raw, found


# ---------------------------------------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------------------------------------
def eeg_pool(path, cfg_entry):
    """The EEG channels of tool 2's configuration (harmonized names) that the median channel is taken from: the remap
    without its EOG names, the channels declared as context channels, and the online reference (a flat channel in the
    data as recorded)."""
    online, _ = RL.online_reference(path)
    context = set((cfg_entry.get('context_channels') or {}).values())
    return [ch for orig, ch in cfg_entry.get('remap', {}).items()
            if orig not in context and orig != online and not RL.EOG_NAME_RE.search(ch)]


def channel_quality_warnings(path, cfg_entry, pool, checked, eog=None):
    """Warnings ([] when there is nothing to say) on the channels a model receives, read as recorded.

    pool = the EEG channels of tool 2's configuration (harmonized names): the median of the comparison. checked =
    {harmonized EEG name: its role, e.g. 'electrode' or 'reference of C3, F3'}. eog = {label: original name}.
    60 epochs of 30 s spread over the recording, filtered 0.3-35 Hz. An EEG channel is compared with the median EEG
    channel of the same epoch, so a movement that disturbs every channel does not count: noisy when its standard
    deviation exceeds 3 times that median on more than 20 % of the epochs, dead under 0.1 times. With fewer than 3 EEG
    channels there is no median: only a flat channel is said. The EOG, which eye movements make unlike the EEG: noisy
    when its median standard deviation exceeds 100 uV, flat under 1 uV on more than 20 % of the epochs, after the
    correction set in tool 2 (edf_io_lib.context_gain)."""
    eog = eog or {}
    inverse = {target: orig for orig, target in cfg_entry.get('remap', {}).items()}
    eeg = [ch for ch in dict.fromkeys(list(pool) + list(checked)) if ch in inverse]
    raw, found = _read(path, [inverse[ch] for ch in eeg] + list(eog.values()))
    epochs = sampled_epochs(raw.n_times, raw.info['sfreq'])
    columns = [(ch, found[inverse[ch]]) for ch in eeg if inverse[ch] in found]
    columns += [(label, found[orig]) for label, orig in eog.items() if orig in found]
    if not len(epochs) or not columns:
        return []
    stds = epoch_std_uV(raw, [name for _, name in columns], epochs)
    index = {key: i for i, (key, _) in enumerate(columns)}
    divisors = context_divisors(cfg_entry)
    for label, orig in eog.items():
        if label in index and orig in divisors:
            stds[:, index[label]] /= divisors[orig]
    n = len(epochs)
    shares = compared_with_median(stds, [index[ch] for ch in pool if ch in index])

    warnings = []
    for ch, role in checked.items():
        if ch not in index:
            continue
        i = index[ch]
        flat = (stds[:, i] < FLAT_UV).mean()
        if shares is not None and shares[0][i] > SHARE:
            warnings.append(f'{ch} ({role}): noisy on {shares[0][i]:.0%} of {n} epochs spread over the recording '
                            f'(standard deviation over {NOISY_RATIO:g} times the median EEG channel)')
        elif shares is not None and shares[1][i] > SHARE:
            warnings.append(f'{ch} ({role}): nearly flat on {shares[1][i]:.0%} of {n} epochs spread over the recording '
                            f'(standard deviation under {DEAD_RATIO:g} times the median EEG channel): disconnected?')
        elif flat > SHARE:
            warnings.append(f'{ch} ({role}): flat on {flat:.0%} of {n} epochs spread over the recording '
                            f'(standard deviation under {FLAT_UV:g} uV): disconnected?')
    for label, orig in eog.items():
        if label not in index:
            continue
        sd = stds[:, index[label]]
        after = divided_text(divisors.get(orig, 1.0))
        if np.median(sd) > EOG_MAX_UV:
            warnings.append(f'{label} ({orig}): median standard deviation of {np.median(sd):.0f} uV over {n} epochs{after} '
                            f'(an EOG stays under {EOG_MAX_UV:g} uV): noisy')
        elif (sd < FLAT_UV).mean() > SHARE:
            warnings.append(f'{label} ({orig}): flat on {(sd < FLAT_UV).mean():.0%} of {n} epochs{after} (standard '
                            f'deviation under {FLAT_UV:g} uV): disconnected?')
    return warnings


def context_amplitude(path, orig, role, divide_by=1.0):
    """The amplitude of one context channel `orig` (`role` 'ecg' or 'emg') on 30 epochs spread over the recording, the
    signal divided by `divide_by` first (tool 2's correction): {'ecg_mV': median R wave peak to peak, 'rate': heart rate
    per minute} or {'emg_uV': median RMS 10-100 Hz without the mains}, with 'mains_share' and 'mains'. None when the
    channel is not in the recording or the recording is shorter than one epoch."""
    try:
        raw, found = _read(path, [orig])
    except ValueError:          # none of the channels asked for in the file (the BrainVision reader says so)
        return None
    if orig not in found:
        return None
    sfreq = raw.info['sfreq']
    n = int(round(EPOCH_S * sfreq))
    epochs = sampled_epochs(raw.n_times, sfreq, N_EPOCHS_CONTEXT)
    if not len(epochs):
        return None
    sigs = [raw.get_data(picks=[found[orig]], start=int(e) * n, stop=int(e) * n + n)[0] * 1e6 / divide_by
            for e in epochs]
    if role == 'ecg':
        values = np.array([ecg_amplitude_uV(s, sfreq) for s in sigs], dtype=float)
        if np.isnan(values[:, 1]).all():
            measures = {'ecg_mV': np.nan, 'rate': np.nan}
        else:
            measures = {'ecg_mV': np.nanmedian(values[:, 0]) / 1000, 'rate': np.nanmedian(values[:, 1])}
    else:
        measures = {'emg_uV': float(np.median([emg_rms_uV(s, sfreq) for s in sigs]))}
    measures['mains_share'], measures['mains'] = mains_share(np.concatenate(sigs), sfreq)
    return measures


def context_amplitude_warnings(path, cfg_entry, roles=('ecg', 'emg')):
    """Warnings ([] when there is nothing to say) on tool 2's context channels `roles` ('ecg', 'emg'), from 30 epochs
    spread over the recording, after the correction set in tool 2 (edf_io_lib.context_gain). ECG: R waves outside 0.1-5
    mV peak to peak, or no heart rate in 30-120 per minute (then it does not look like an ECG). Chin EMG: RMS 10-100 Hz
    without the mains over 100 uV. Both: the mains holding more than half of the power. An amplitude out of range comes
    with its factor to the typical value (1 mV, 5 uV), rounded to a power of ten: an order of magnitude, to be confirmed
    on the hardware."""
    ctx = cfg_entry.get('context_channels') or {}
    wanted = {role: ctx.get(role) for role in roles if ctx.get(role)}
    if not wanted:
        return []
    gains = context_gain(cfg_entry)
    warnings = []
    for role, orig in wanted.items():
        measures = context_amplitude(path, orig, role, gains.get(role, 1.0))
        if measures is None:
            continue
        name = 'ECG' if role == 'ecg' else 'chin EMG'
        after = divided_text(gains.get(role, 1.0))
        if role == 'ecg':
            amplitude_mV, rate = measures['ecg_mV'], measures['rate']
            if np.isnan(rate) or not HEART_RATE_RANGE[0] <= rate <= HEART_RATE_RANGE[1]:
                found_rate = 'no regular beat found' if np.isnan(rate) else f'heart rate {rate:.0f} per minute'
                warnings.append(f'{name} ({orig}): does not look like an ECG ({found_rate})')
            elif not ECG_RANGE_MV[0] <= amplitude_mV <= ECG_RANGE_MV[1]:
                warnings.append(f'{name} ({orig}): R waves of {amplitude_mV:.3g} mV peak to peak{after}, about '
                                f'{decade(amplitude_mV / ECG_TYPICAL_MV):g} times the usual {ECG_TYPICAL_MV:g} mV '
                                f'(expected {ECG_RANGE_MV[0]:g}-{ECG_RANGE_MV[1]:g} mV): the gain of its input may '
                                'differ from the one the file declares (order of magnitude, to confirm on the hardware)')
        else:
            rms = measures['emg_uV']
            if rms > EMG_MAX_UV:
                warnings.append(f'{name} ({orig}): RMS of {rms:.0f} uV on 10-100 Hz{after}, about '
                                f'{decade(rms / EMG_TYPICAL_UV):g} times the usual {EMG_TYPICAL_UV:g} uV (expected up to '
                                f'{EMG_MAX_UV:g} uV): the gain of its input may differ from the one the file declares '
                                '(order of magnitude, to confirm on the hardware)')
        if measures['mains_share'] > MAINS_SHARE:
            warnings.append(f'{name} ({orig}): the mains ({measures["mains"]} Hz) holds {measures["mains_share"]:.0%} '
                            'of its power on 1-100 Hz')
    return warnings


def context_gain_advice(role, label, orig, measures, divide_by=1.0):
    """Tool 2's line on the amplitude of one context channel of a channel configuration: (kind, text), kind 'ok', 'warn'
    or 'info' (ui_lib.msg). `measures` = context_amplitude of the first recordings of the configuration, read as recorded
    (None for a recording without the channel), `divide_by` = the value of its 'Divide by' field. Out of the expected
    range, it proposes the factor that brings the median recording to the usual amplitude (a power of ten, so the gain
    of an adapter such as BIP2AUX, 100, comes out as it is); it never sets the field itself."""
    if role == 'ecg':
        values = [m['ecg_mV'] / divide_by for m in measures if m is not None and np.isfinite(m['ecg_mV'])]
        unit, expected, typical, what = 'mV', ECG_RANGE_MV, ECG_TYPICAL_MV, 'R waves of'
        usual = f'{ECG_RANGE_MV[0]:g}-{ECG_RANGE_MV[1]:g} mV'
    else:
        values = [m['emg_uV'] / divide_by for m in measures if m is not None and np.isfinite(m['emg_uV'])]
        unit, expected, typical, what = 'uV', (0.0, EMG_MAX_UV), EMG_TYPICAL_UV, 'RMS of'
        usual = f'up to {EMG_MAX_UV:g} uV'
    n = sum(1 for m in measures if m is not None)
    if not values:
        reason = 'no regular beat found' if role == 'ecg' and n else 'not in these recordings'
        return 'info', f'{label} ({orig}): amplitude not checked ({reason}).'
    lo, hi = min(values), max(values)
    span = f'{lo:.3g}' if np.isclose(lo, hi, rtol=0.05) else f'{lo:.3g}-{hi:.3g}'
    divided = f' divided by {divide_by:g}' if divide_by != 1 else ''
    out = [v for v in values if not expected[0] <= v <= expected[1]]
    if not out:
        return 'ok', f'{label} ({orig}){divided}: {what} {span} {unit} in {len(values)} recording(s) (usual: {usual}).'
    factor = decade(float(np.median(values)) / typical)
    if factor == 1:            # out of range, yet not a power of ten away from the usual value: no factor to propose
        return 'warn', (f'{label} ({orig}){divided}: {what} {span} {unit} in {len(out)}/{len(values)} recording(s) '
                        f'(usual: {usual}).')
    return 'warn', (f'{label} ({orig}){divided}: {what} {span} {unit} in {len(out)}/{len(values)} recording(s), about '
                    f'{factor:g} times the usual {typical:g} {unit}: its input may have a gain the file does not declare '
                    f'(an adapter such as BIP2AUX amplifies by 100). Set Divide by to {divide_by * factor:g}?')
