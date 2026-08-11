import numpy as np

from processing import (
    crop_up_intervals,
    compute_refractory_period,
    count_spikes_per_upstate,
    pulse_to_event_latencies,
    pulse_to_up_latencies,
    _mwu_stats,
    compute_refractory_any_to_type,
    _clip_events_to_bounds,
    _check_peak_indices,
    _p_to_sig_label,
)


def test_crop_up_intervals_basic_crop():
    # One UP-state from sample 0 to sample 2000 (dt=0.001s -> 2s long).
    # start_s=0.3 -> 300 samples offset, end_s=1.0 -> 1000 samples offset.
    UP = [0]
    DOWN = [2000]
    cropped_UP, cropped_DOWN = crop_up_intervals(UP, DOWN, dt=0.001, start_s=0.3, end_s=1.0)

    np.testing.assert_array_equal(cropped_UP, [300])
    np.testing.assert_array_equal(cropped_DOWN, [1000])


def test_crop_up_intervals_clamped_to_down():
    # DOWN comes before the requested end_s -> crop must stop at DOWN, not overshoot it.
    UP = [0]
    DOWN = [500]
    cropped_UP, cropped_DOWN = crop_up_intervals(UP, DOWN, dt=0.001, start_s=0.3, end_s=1.0)

    np.testing.assert_array_equal(cropped_UP, [300])
    np.testing.assert_array_equal(cropped_DOWN, [500])


def test_crop_up_intervals_skips_too_short_interval():
    # UP-state ends before start_s has even elapsed -> nothing left to keep, interval is dropped.
    UP = [0]
    DOWN = [250]
    cropped_UP, cropped_DOWN = crop_up_intervals(UP, DOWN, dt=0.001, start_s=0.3, end_s=1.0)

    assert cropped_UP.size == 0
    assert cropped_DOWN.size == 0


def test_crop_up_intervals_multiple_intervals():
    UP = [0, 5000]
    DOWN = [2000, 6000]
    cropped_UP, cropped_DOWN = crop_up_intervals(UP, DOWN, dt=0.001, start_s=0.3, end_s=1.0)

    np.testing.assert_array_equal(cropped_UP, [300, 5300])
    np.testing.assert_array_equal(cropped_DOWN, [1000, 6000])


def test_crop_up_intervals_empty_input():
    cropped_UP, cropped_DOWN = crop_up_intervals([], [], dt=0.001)

    assert cropped_UP.size == 0
    assert cropped_DOWN.size == 0


def test_crop_up_intervals_mismatched_lengths_uses_shorter():
    # UP has one more entry than DOWN -> the dangling UP without a matching DOWN is ignored.
    UP = [0, 5000]
    DOWN = [2000]
    cropped_UP, cropped_DOWN = crop_up_intervals(UP, DOWN, dt=0.001, start_s=0.3, end_s=1.0)

    np.testing.assert_array_equal(cropped_UP, [300])
    np.testing.assert_array_equal(cropped_DOWN, [1000])


def test_compute_refractory_period_basic_gap():
    # time_s[i] = i * 0.01s. UP-state 1: samples 10..20. UP-state 2: samples 50..60.
    # Refractory time = time at next UP-onset (0.50s) minus time at previous DOWN (0.20s) = 0.30s.
    time_s = np.arange(100) * 0.01
    UP = [10, 50]
    DOWN = [20, 60]

    refrac = compute_refractory_period(UP, DOWN, time_s)

    np.testing.assert_allclose(refrac, [0.30])


def test_compute_refractory_period_three_states_two_gaps():
    time_s = np.arange(100) * 0.01
    UP = [10, 50, 80]
    DOWN = [20, 60, 95]

    refrac = compute_refractory_period(UP, DOWN, time_s)

    # gap 1: 0.50 - 0.20 = 0.30 ; gap 2: 0.80 - 0.60 = 0.20
    np.testing.assert_allclose(refrac, [0.30, 0.20])


def test_compute_refractory_period_needs_at_least_two_up_states():
    time_s = np.arange(100) * 0.01
    UP = [10]
    DOWN = [20]

    refrac = compute_refractory_period(UP, DOWN, time_s)

    assert refrac.size == 0


def test_compute_refractory_period_drops_invalid_pairs():
    # Middle pair is invalid (DOWN before UP) and must be filtered out before pairing up gaps.
    time_s = np.arange(100) * 0.01
    UP = [10, 5, 60]
    DOWN = [20, 2, 70]

    refrac = compute_refractory_period(UP, DOWN, time_s)

    np.testing.assert_allclose(refrac, [0.40])  # time_s[60] - time_s[20] = 0.60 - 0.20


def test_compute_refractory_period_sorts_unsorted_input():
    # UP/DOWN given out of chronological order must still yield the correct gap.
    time_s = np.arange(100) * 0.01
    UP = [50, 10]
    DOWN = [60, 20]

    refrac = compute_refractory_period(UP, DOWN, time_s)

    np.testing.assert_allclose(refrac, [0.30])


def test_compute_refractory_period_drops_negative_gaps():
    # Next UP starts before the previous DOWN's timestamp (overlapping states) -> negative
    # refractory time is nonsensical and must be discarded, not returned as-is.
    time_s = np.arange(100) * 0.01
    UP = [10, 15]
    DOWN = [20, 25]

    refrac = compute_refractory_period(UP, DOWN, time_s)

    assert refrac.size == 0


def test_compute_refractory_period_too_short_time_vector():
    refrac = compute_refractory_period([0, 1], [1, 2], time_s=[0.0])

    assert refrac.size == 0


def test_count_spikes_per_upstate_basic_count():
    # UP-state: samples 10..20 -> t=[0.10, 0.20). One spike sits right on the boundary at 0.20
    # and must NOT be counted (half-open interval), the other three are inside.
    time_s = np.arange(100) * 0.01
    up_idx = [10]
    down_idx = [20]
    spike_times_s = [0.10, 0.15, 0.19, 0.20, 0.25]

    counts, rates_hz, durations_s = count_spikes_per_upstate(spike_times_s, up_idx, down_idx, time_s)

    np.testing.assert_array_equal(counts, [3])
    np.testing.assert_allclose(durations_s, [0.10])
    np.testing.assert_allclose(rates_hz, [30.0])  # 3 spikes / 0.10s


def test_count_spikes_per_upstate_spike_at_start_is_included():
    time_s = np.arange(100) * 0.01
    counts, _, _ = count_spikes_per_upstate([0.10], [10], [20], time_s)

    np.testing.assert_array_equal(counts, [1])


def test_count_spikes_per_upstate_multiple_upstates():
    time_s = np.arange(100) * 0.01
    up_idx = [10, 50]
    down_idx = [20, 60]
    spike_times_s = [0.12, 0.55, 0.58]

    counts, rates_hz, durations_s = count_spikes_per_upstate(spike_times_s, up_idx, down_idx, time_s)

    np.testing.assert_array_equal(counts, [1, 2])
    np.testing.assert_allclose(durations_s, [0.10, 0.10])
    np.testing.assert_allclose(rates_hz, [10.0, 20.0])


def test_count_spikes_per_upstate_nonpositive_duration_is_skipped():
    # UP and DOWN point at the same sample -> zero duration -> stays NaN/0, not counted.
    time_s = np.arange(100) * 0.01
    counts, rates_hz, durations_s = count_spikes_per_upstate([0.30], [30], [30], time_s)

    np.testing.assert_array_equal(counts, [0])
    assert np.isnan(durations_s[0])
    assert np.isnan(rates_hz[0])


def test_count_spikes_per_upstate_empty_upstates():
    time_s = np.arange(100) * 0.01
    counts, rates_hz, durations_s = count_spikes_per_upstate([0.1], [], [], time_s)

    assert counts.size == 0
    assert rates_hz.size == 0
    assert durations_s.size == 0


def test_count_spikes_per_upstate_no_spikes_at_all_leaves_rate_nan():
    # No spike train given at all (e.g. no MUA detected) is different from "0 spikes found in
    # this particular window": duration is still computed, but rate stays NaN, not 0.0.
    time_s = np.arange(100) * 0.01
    counts, rates_hz, durations_s = count_spikes_per_upstate([], [10], [20], time_s)

    np.testing.assert_array_equal(counts, [0])
    np.testing.assert_allclose(durations_s, [0.10])
    assert np.isnan(rates_hz[0])


def test_count_spikes_per_upstate_out_of_bounds_index_is_clipped():
    # down_idx points past the end of time_s -> must be clipped to the last valid sample,
    # not raise an IndexError.
    time_s = np.arange(100) * 0.01
    counts, rates_hz, durations_s = count_spikes_per_upstate([0.96], [95], [150], time_s)

    np.testing.assert_array_equal(counts, [1])
    np.testing.assert_allclose(durations_s, [time_s[99] - time_s[95]])


def test_pulse_to_event_latencies_basic():
    time_s = np.arange(100) * 0.01
    pulse_times = [0.10]
    event_indices = [20]  # time_s[20] = 0.20

    latencies = pulse_to_event_latencies(pulse_times, event_indices, time_s)

    np.testing.assert_allclose(latencies, [0.10])


def test_pulse_to_event_latencies_uses_closest_preceding_pulse():
    time_s = np.arange(100) * 0.01
    pulse_times = [0.05, 0.15]  # two pulses before the event, sorted ascending
    event_indices = [20]  # t = 0.20

    latencies = pulse_to_event_latencies(pulse_times, event_indices, time_s)

    np.testing.assert_allclose(latencies, [0.05])  # 0.20 - 0.15, not 0.20 - 0.05


def test_pulse_to_event_latencies_excludes_events_beyond_max_win_s():
    time_s = np.arange(100) * 0.01
    pulse_times = [0.10]
    event_indices = [20]  # latency would be 0.10

    latencies = pulse_to_event_latencies(pulse_times, event_indices, time_s, max_win_s=0.05)

    assert latencies.size == 0


def test_pulse_to_event_latencies_skips_events_with_no_preceding_pulse():
    time_s = np.arange(100) * 0.01
    pulse_times = [0.50]
    # First event (t=0.10) has no pulse before it -> skipped. Second event (t=0.60) does.
    event_indices = [10, 60]

    latencies = pulse_to_event_latencies(pulse_times, event_indices, time_s)

    np.testing.assert_allclose(latencies, [0.10])  # only the second event's latency


def test_pulse_to_event_latencies_empty_pulses_returns_empty():
    time_s = np.arange(100) * 0.01
    latencies = pulse_to_event_latencies([], [10, 20], time_s)

    assert latencies.size == 0


def test_pulse_to_event_latencies_none_pulses_returns_empty():
    time_s = np.arange(100) * 0.01
    latencies = pulse_to_event_latencies(None, [10, 20], time_s)

    assert latencies.size == 0


def test_pulse_to_event_latencies_empty_events_returns_empty():
    time_s = np.arange(100) * 0.01
    latencies = pulse_to_event_latencies([0.1], [], time_s)

    assert latencies.size == 0


def test_pulse_to_up_latencies_wraps_event_latencies():
    # Documented as a backwards-compat wrapper -> must behave identically to the function it wraps.
    time_s = np.arange(100) * 0.01
    pulse_times = [0.10]
    up_indices = [20]

    wrapped = pulse_to_up_latencies(pulse_times, up_indices, time_s)
    direct = pulse_to_event_latencies(pulse_times, up_indices, time_s)

    np.testing.assert_allclose(wrapped, direct)


def test_mwu_stats_too_few_samples_spont_side():
    # Only 1 spontaneous value -> below the n>=2 minimum, must bail out with NaNs (not crash).
    out = _mwu_stats(spont=[1.0], trig=[1.0, 2.0, 3.0])

    assert out["n_sp"] == 1
    assert out["n_tr"] == 3
    assert np.isnan(out["p"])
    assert np.isnan(out["delta"])
    assert out["significant"] is False


def test_mwu_stats_too_few_samples_trig_side():
    out = _mwu_stats(spont=[1.0, 2.0, 3.0], trig=[])

    assert out["n_sp"] == 3
    assert out["n_tr"] == 0
    assert np.isnan(out["p"])
    assert np.isnan(out["delta"])
    assert out["significant"] is False


def test_mwu_stats_filters_nonfinite_values_before_counting():
    # NaN/inf entries must not count towards n_sp/n_tr or be fed into the test.
    spont = [1.0, 2.0, np.nan, np.inf, 3.0]
    trig = [4.0, 5.0, 6.0]

    out = _mwu_stats(spont, trig)

    assert out["n_sp"] == 3  # only 1.0, 2.0, 3.0 are finite
    assert out["n_tr"] == 3


def test_mwu_stats_fully_separated_groups():
    # Every spont value < every trig value -> Cliff's delta must be exactly -1.0,
    # and the groups are clearly significantly different.
    spont = [1, 2, 3, 4, 5]
    trig = [10, 11, 12, 13, 14]

    out = _mwu_stats(spont, trig, alpha=0.05)

    assert out["delta"] == -1.0
    np.testing.assert_allclose(out["p"], 0.007936507936507936)
    assert out["significant"] is True


def test_mwu_stats_identical_groups_not_significant():
    spont = [1, 2, 3]
    trig = [1, 2, 3]

    out = _mwu_stats(spont, trig)

    assert out["delta"] == 0.0
    np.testing.assert_allclose(out["p"], 1.0)
    assert out["significant"] is False


def test_mwu_stats_alpha_threshold_controls_significant_flag():
    # Same data (p ~ 0.0079) but a stricter alpha must flip "significant" to False.
    spont = [1, 2, 3, 4, 5]
    trig = [10, 11, 12, 13, 14]

    lenient = _mwu_stats(spont, trig, alpha=0.05)
    strict = _mwu_stats(spont, trig, alpha=0.001)

    assert lenient["significant"] is True
    assert strict["significant"] is False


def test_refractory_any_to_type_two_spontaneous_states():
    time_s = np.arange(100) * 0.01

    refrac_spont, refrac_trig = compute_refractory_any_to_type(
        Spontaneous_UP=[10, 50], Spontaneous_DOWN=[20, 60],
        Pulse_triggered_UP=[], Pulse_triggered_DOWN=[],
        Pulse_associated_UP=[], Pulse_associated_DOWN=[],
        time_s=time_s,
    )

    np.testing.assert_allclose(refrac_spont, [0.30])  # time_s[50]-time_s[20]
    assert refrac_trig.size == 0


def test_refractory_any_to_type_buckets_by_current_states_type():
    # spont -> trig transition: the gap must land in refrac_any_to_trig, not refrac_any_to_spont.
    time_s = np.arange(100) * 0.01

    refrac_spont, refrac_trig = compute_refractory_any_to_type(
        Spontaneous_UP=[10], Spontaneous_DOWN=[20],
        Pulse_triggered_UP=[50], Pulse_triggered_DOWN=[60],
        Pulse_associated_UP=[], Pulse_associated_DOWN=[],
        time_s=time_s,
    )

    assert refrac_spont.size == 0
    np.testing.assert_allclose(refrac_trig, [0.30])


def test_refractory_any_to_type_associated_ignored_as_target_but_kept_as_prev():
    # Order by time: spont(10-20) -> assoc(50-60) -> spont(80-90).
    # Gap into the assoc state is dropped entirely; gap out of it still counts using its offset.
    time_s = np.arange(100) * 0.01

    refrac_spont, refrac_trig = compute_refractory_any_to_type(
        Spontaneous_UP=[10, 80], Spontaneous_DOWN=[20, 90],
        Pulse_triggered_UP=[], Pulse_triggered_DOWN=[],
        Pulse_associated_UP=[50], Pulse_associated_DOWN=[60],
        time_s=time_s,
    )

    np.testing.assert_allclose(refrac_spont, [0.20])  # time_s[80]-time_s[60]
    assert refrac_trig.size == 0


def test_refractory_any_to_type_sorts_across_categories_by_time():
    # Spontaneous UP is passed first but occurs LATER in time than the triggered UP ->
    # result must reflect chronological order, not argument order.
    time_s = np.arange(100) * 0.01

    refrac_spont, refrac_trig = compute_refractory_any_to_type(
        Spontaneous_UP=[50], Spontaneous_DOWN=[60],
        Pulse_triggered_UP=[10], Pulse_triggered_DOWN=[20],
        Pulse_associated_UP=[], Pulse_associated_DOWN=[],
        time_s=time_s,
    )

    np.testing.assert_allclose(refrac_spont, [0.30])  # time_s[50]-time_s[20]
    assert refrac_trig.size == 0


def test_refractory_any_to_type_drops_negative_gaps():
    time_s = np.arange(100) * 0.01

    refrac_spont, refrac_trig = compute_refractory_any_to_type(
        Spontaneous_UP=[10], Spontaneous_DOWN=[20],
        Pulse_triggered_UP=[15], Pulse_triggered_DOWN=[25],
        Pulse_associated_UP=[], Pulse_associated_DOWN=[],
        time_s=time_s,
    )

    assert refrac_spont.size == 0
    assert refrac_trig.size == 0


def test_refractory_any_to_type_single_state_has_no_gaps():
    time_s = np.arange(100) * 0.01

    refrac_spont, refrac_trig = compute_refractory_any_to_type(
        Spontaneous_UP=[10], Spontaneous_DOWN=[20],
        Pulse_triggered_UP=[], Pulse_triggered_DOWN=[],
        Pulse_associated_UP=[], Pulse_associated_DOWN=[],
        time_s=time_s,
    )

    assert refrac_spont.size == 0
    assert refrac_trig.size == 0


def test_clip_events_to_bounds_basic_filtering():
    # time_s spans 0.00..0.99, pre_s/post_s=0.10 -> valid window is [0.10, 0.89].
    time_s = np.arange(100) * 0.01
    pulse_times = [0.05, 0.10, 0.5, 0.89, 0.95]

    kept = _clip_events_to_bounds(pulse_times, time_s, pre_s=0.10, post_s=0.10)

    np.testing.assert_allclose(kept, [0.10, 0.5, 0.89])


def test_clip_events_to_bounds_boundaries_are_inclusive():
    time_s = np.arange(100) * 0.01

    kept = _clip_events_to_bounds([0.10, 0.89], time_s, pre_s=0.10, post_s=0.10)

    np.testing.assert_allclose(kept, [0.10, 0.89])


def test_clip_events_to_bounds_window_too_narrow_returns_empty():
    # pre_s + post_s exceeds the recording duration -> hi <= lo -> nothing can pass.
    time_s = np.arange(10) * 0.01  # 0.00..0.09

    kept = _clip_events_to_bounds([0.045], time_s, pre_s=0.05, post_s=0.05)

    assert kept.size == 0


def test_clip_events_to_bounds_none_pulse_times():
    time_s = np.arange(100) * 0.01

    kept = _clip_events_to_bounds(None, time_s, pre_s=0.0, post_s=0.0)

    assert kept.size == 0


def test_clip_events_to_bounds_empty_pulse_times():
    time_s = np.arange(100) * 0.01

    kept = _clip_events_to_bounds([], time_s, pre_s=0.0, post_s=0.0)

    assert kept.size == 0


def test_clip_events_to_bounds_empty_time_s():
    kept = _clip_events_to_bounds([0.1], [], pre_s=0.0, post_s=0.0)

    assert kept.size == 0


def test_check_peak_indices_all_in_bounds(capsys):
    # capsys ("captured system output") lets us read what a test printed to stdout.
    _check_peak_indices("MyLabel", [1, 5, 9], n=10)

    captured = capsys.readouterr()
    assert "count=3" in captured.out
    assert "out_of_bounds=0" in captured.out
    assert "MyLabel" in captured.out


def test_check_peak_indices_counts_out_of_bounds_indices(capsys):
    # n=10 -> valid range is [0, 9]. -1 (too low) and 15, 20 (too high) are out of bounds; 5 is fine.
    _check_peak_indices("MyLabel", [-1, 5, 15, 20], n=10)

    captured = capsys.readouterr()
    assert "count=4" in captured.out
    assert "out_of_bounds=3" in captured.out


def test_check_peak_indices_empty_peaks(capsys):
    _check_peak_indices("Empty", [], n=10)

    captured = capsys.readouterr()
    assert "count=0" in captured.out
    assert "out_of_bounds=0" in captured.out


def test_check_peak_indices_exact_message_format(capsys):
    _check_peak_indices("Spon_Peaks", [1, 2], n=5)

    captured = capsys.readouterr()
    assert captured.out.strip() == "[DIAG] Spon_Peaks: count=2, out_of_bounds=0 (n_time=5)"


def test_check_peak_indices_returns_none():
    result = _check_peak_indices("Label", [1, 2, 3], n=10)

    assert result is None


def test_p_to_sig_label_nonfinite_is_not_available():
    assert _p_to_sig_label(np.nan) == "n/a"
    assert _p_to_sig_label(np.inf) == "n/a"


def test_p_to_sig_label_significance_tiers():
    assert _p_to_sig_label(1e-5) == "****"
    assert _p_to_sig_label(5e-4) == "***"
    assert _p_to_sig_label(5e-3) == "**"
    assert _p_to_sig_label(0.03) == "*"
    assert _p_to_sig_label(0.5) == "n.s."


def test_p_to_sig_label_thresholds_are_exclusive():
    # Comparisons use strict "<", so a p-value exactly AT a threshold falls into the
    # *less* significant bucket, not the more significant one.
    assert _p_to_sig_label(1e-4) == "***"   # not "****"
    assert _p_to_sig_label(1e-3) == "**"    # not "***"
    assert _p_to_sig_label(1e-2) == "*"     # not "**"
    assert _p_to_sig_label(0.05) == "n.s."  # not "*" (default alpha=0.05)


def test_p_to_sig_label_custom_alpha_shifts_star_cutoff():
    # p=0.08 is "n.s." at the default alpha=0.05, but becomes "*" once alpha is relaxed to 0.1.
    assert _p_to_sig_label(0.08, alpha=0.05) == "n.s."
    assert _p_to_sig_label(0.08, alpha=0.1) == "*"


def test_refractory_any_to_type_all_empty():
    time_s = np.arange(100) * 0.01

    refrac_spont, refrac_trig = compute_refractory_any_to_type(
        Spontaneous_UP=[], Spontaneous_DOWN=[],
        Pulse_triggered_UP=[], Pulse_triggered_DOWN=[],
        Pulse_associated_UP=[], Pulse_associated_DOWN=[],
        time_s=time_s,
    )

    assert refrac_spont.size == 0
    assert refrac_trig.size == 0
