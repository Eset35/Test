import numpy as np
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import csd


"""
This module contain reusable functions that are called from various modules.
"""


# Setup frequency ranges for bands (in Hz)
theta_range = (4, 8)
alpha_range = (8, 12)
mu_range = (12, 16)
beta_range = (16, 23)


def calc_metrics(x, freq, selected_metrics=None):
    """
    Calculate metrics for given chunk of 1D-data

    Parameters
    ----------
    x : array_like
        Array containing numbers.
    freq : float
        Sampling frequency
    selected_features : array, optional
        Array of features to return.
        Now all features are calculated but not all of them are returned.

    Returns
    -------
    features_to_return : dict
        Dictionary that contain names of features and their calculated values.
    """

    mean = np.mean(x)
    sd = np.std(x)
    skewness = skew(x)
    kurt = kurtosis(x)
    min_ = np.min(x)
    max_ = np.max(x)
    median_abs = np.median(np.abs(x))

    bin_edges = np.linspace(min_, max_, 101)
    amp_distribution, _ = np.histogram(x, bins=bin_edges)
    amp_distribution = amp_distribution / np.sum(amp_distribution)
    amp_entropy = entropy(amp_distribution)

    freqs, ps = calc_spectrum(x, freq)
    psd_distribution = ps / np.sum(ps)
    psd_entropy = entropy(psd_distribution)

    tp, ap, mp, bp = calc_power(freqs, ps)
    dominant_freq = freqs[np.argmax(ps)]

    n_roots, std_min, std_max, slope_min, slope_max = roots(x, freq)

    # metrics = {'mean': mean,
    #            'sd': sd,
    #            'skewness': skewness,
    #            'kurtosis': kurt,
    #            'min': min_,
    #            'max': max_,
    #            'median_abs': median_abs,
    #            'amp_entropy': amp_entropy,
    #            'psd_entropy': psd_entropy,
    #            'theta_power': tp,
    #            'alpha_power': ap,
    #            'mu_power': mp,
    #            'beta_power': bp,
    #            'dominant_freq': dominant_freq,
    #            'a2t_power': ap / tp,
    #            'm2t_power': mp / tp,
    #            'b2t_power': bp / tp,
    #            'm2a_power': bp / ap,
    #            'b2a_power': tp / ap,
    #            'b2m_power': bp / mp}

    metrics = {'mean': mean,
               'sd': np.log(sd),
               'skewness': skewness,
               'kurtosis': kurt,
               'min': np.log(np.abs(min_)),
               'max': np.log(np.abs(max_)),
               'median_abs': np.log(median_abs),
               'amp_entropy': amp_entropy,
               'psd_entropy': psd_entropy,
               'theta_power': np.log10(tp),
               'alpha_power': np.log10(ap),
               'mu_power': np.log10(mp),
               'beta_power': np.log10(bp),
               'dominant_freq': dominant_freq,
               'a2t_power': ap / tp,
               'm2t_power': mp / tp,
               'b2t_power': bp / tp,
               'm2a_power': bp / ap,
               'b2a_power': tp / ap,
               'b2m_power': bp / mp,
               'n_roots': n_roots,
               'std_min': std_min,
               'std_max': std_max,
               'slope_min': slope_min,
               'slope_max': slope_max}

    if not selected_metrics:
        selected_metrics = ['mean',
                            'sd',
                            'skewness',
                            'kurtosis',
                            'min',
                            'max',
                            'median_abs']

    metrics_to_return = dict()
    for k in selected_metrics:
        metrics_to_return[k] = metrics[k]
    return metrics_to_return


def calc_cross_metrics(data, ch_names, freq, selected_metrics=None):
    """
    Calculate cross-metrics for given chunk of data for several channels

    Parameters
    ----------
    data : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    ch_names : array_like
        Names of channels.
    freq : float
        Sampling frequency
    selected_metrics : array, optional
        Array of features to return.
        Now all features are calculated but not all of them are returned.

    Returns
    -------
    features_to_return : dict
        Dictionary that contain names of features and their calculated values.
    """

    # Calculate number of points for FFT (next power of 2)
    nfft = int(2**(np.floor(np.log2(data.shape[1])) + 3))

    # Create empty arrays for data
    g = np.zeros((data.shape[0], data.shape[0], int(nfft / 2 + 1)), dtype=complex)
    coherence_theta = np.zeros((data.shape[0], data.shape[0]))
    coherence_alpha = np.zeros((data.shape[0], data.shape[0]))
    coherence_mu = np.zeros((data.shape[0], data.shape[0]))
    coherence_beta = np.zeros((data.shape[0], data.shape[0]))
    phase_theta = np.zeros((data.shape[0], data.shape[0]))
    phase_alpha = np.zeros((data.shape[0], data.shape[0]))
    phase_mu = np.zeros((data.shape[0], data.shape[0]))
    phase_beta = np.zeros((data.shape[0], data.shape[0]))

    # Calculate cross-spectral densities between channels
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i >= j:
                freqs, g[i, j, :] = csd(data[i], data[j], fs=freq, nperseg=int(data.shape[1] / 2), nfft=nfft)

    # Calculate edges of bands
    theta_idx = [np.argmin(np.abs(freqs - theta_range[0])), np.argmin(np.abs(freqs - theta_range[1]))]
    alpha_idx = [np.argmin(np.abs(freqs - alpha_range[0])), np.argmin(np.abs(freqs - alpha_range[1]))]
    mu_idx = [np.argmin(np.abs(freqs - mu_range[0])), np.argmin(np.abs(freqs - mu_range[1]))]
    beta_idx = [np.argmin(np.abs(freqs - beta_range[0])), np.argmin(np.abs(freqs - beta_range[1]))]

    # Calculate cross-metrics
    header = []
    metrics = []
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if (ch_names[i] == 'CZ') or (ch_names[j] == 'CZ'):
                continue
            if i > j:
                # calculate coherence and phase for whole spectrum
                coherence_i_j = np.real(np.abs(g[i, j, :])**2 / (g[i, i, :] * g[j, j, :]))
                phase_i_j = np.angle(g[i, j, :], deg=True)
                phase_i_j[phase_i_j < 0] += 360

                # calculate coherence and phase for individual bands
                coherence_theta[i, j] = np.median(coherence_i_j[theta_idx[0]:theta_idx[1]])
                coherence_alpha[i, j] = np.median(coherence_i_j[alpha_idx[0]:alpha_idx[1]])
                coherence_mu[i, j] = np.median(coherence_i_j[mu_idx[0]:mu_idx[1]])
                coherence_beta[i, j] = np.median(coherence_i_j[beta_idx[0]:beta_idx[1]])

                phase_theta[i, j] = np.mean(phase_i_j[theta_idx[0]:theta_idx[1]])
                phase_alpha[i, j] = np.mean(phase_i_j[alpha_idx[0]:alpha_idx[1]])
                phase_mu[i, j] = np.mean(phase_i_j[mu_idx[0]:mu_idx[1]])
                phase_beta[i, j] = np.mean(phase_i_j[beta_idx[0]:beta_idx[1]])

                header += [f'coherence_th_{ch_names[j]}_{ch_names[i]}']
                header += [f'coherence_al_{ch_names[j]}_{ch_names[i]}']
                header += [f'coherence_mu_{ch_names[j]}_{ch_names[i]}']
                header += [f'coherence_be_{ch_names[j]}_{ch_names[i]}']

                header += [f'phase_th_{ch_names[j]}_{ch_names[i]}']
                header += [f'phase_al_{ch_names[j]}_{ch_names[i]}']
                header += [f'phase_mu_{ch_names[j]}_{ch_names[i]}']
                header += [f'phase_be_{ch_names[j]}_{ch_names[i]}']

                metrics += [coherence_theta[i, j],
                            coherence_alpha[i, j],
                            coherence_mu[i, j],
                            coherence_beta[i, j]]
                metrics += [phase_theta[i, j],
                            phase_alpha[i, j],
                            phase_mu[i, j],
                            phase_beta[i, j]]

                # calculate difference in mean, min and max values
                mean_diff = np.mean(data[j]) - np.mean(data[i])
                min_diff = np.min(data[j]) - np.min(data[i])
                max_diff = np.max(data[j]) - np.max(data[i])

                header += [f'mean_diff_{ch_names[j]}_{ch_names[i]}']
                header += [f'min_diff_{ch_names[j]}_{ch_names[i]}']
                header += [f'max_diff_{ch_names[j]}_{ch_names[i]}']

                metrics += [mean_diff, min_diff, max_diff]

                # calculate ratio for mean, min and max values
                mean_ratio = np.log(np.abs(np.mean(data[j]) / np.mean(data[i])))
                min_ratio = np.log(np.abs(np.min(data[j]) / np.min(data[i])))
                max_ratio = np.log(np.abs(np.max(data[j]) / np.max(data[i])))

                header += [f'mean_ratio_{ch_names[j]}_{ch_names[i]}']
                header += [f'min_ratio_{ch_names[j]}_{ch_names[i]}']
                header += [f'max_ratio_{ch_names[j]}_{ch_names[i]}']

                metrics += [mean_ratio, min_ratio, max_ratio]

    metrics_dict = dict(zip(header, metrics))
    if not selected_metrics:
        selected_metrics = ['coherence',
                            'mean_diff',
                            'min_diff',
                            'max_diff',
                            'mean_ratio',
                            'min_ratio',
                            'max_ratio']
    metrics_to_return = dict()
    for k in metrics_dict:
        for m in selected_metrics:
            if m in k:
                metrics_to_return[k] = metrics_dict[k]
    return metrics_to_return


def calc_power(freqs, ps):
    """
    Calculate how much power in signal at theta, alpha, mu and beta rhythm bands.

    Parameters
    ----------
    x: numpy array, shape (n_times)
        Containing multi-channels signal.
    freq: int, float
        Sampling frequency

    Returns
    -------
    power: numpy array, shape (3)
        Calculated power for each channel for theta (4-8 Hz), alpha (8-12 Hz),
        mu (12-16 Hz) and beta (16-23 Hz) bands
    """
    df = freqs[1] - freqs[0]

    theta_idx0 = np.argmin(np.abs(freqs - theta_range[0]))
    theta_idx1 = np.argmin(np.abs(freqs - theta_range[1]))
    alpha_idx0 = np.argmin(np.abs(freqs - alpha_range[0]))
    alpha_idx1 = np.argmin(np.abs(freqs - alpha_range[1]))
    mu_idx0 = np.argmin(np.abs(freqs - mu_range[0]))
    mu_idx1 = np.argmin(np.abs(freqs - mu_range[1]))
    beta_idx0 = np.argmin(np.abs(freqs - beta_range[0]))
    beta_idx1 = np.argmin(np.abs(freqs - beta_range[1]))

    theta_power = np.sum(ps[theta_idx0:theta_idx1]) * df * (theta_idx1 - theta_idx0)
    alpha_power = np.sum(ps[alpha_idx0:alpha_idx1]) * df * (alpha_idx1 - alpha_idx0)
    mu_power = np.sum(ps[mu_idx0:mu_idx1]) * df * (mu_idx1 - mu_idx0)
    beta_power = np.sum(ps[beta_idx0:beta_idx1]) * df * (beta_idx1 - beta_idx0)

    return np.array([theta_power, alpha_power, mu_power, beta_power])


def calc_spectrum(x, freq):
    """
    Find dominant positive frequency in power spectrum of signal x.

    Parameters
    ----------
    x : numpy array, shape (n_times)
        Containing multi-channels signal.
    freq : int, float
        Sampling frequency

    Returns
    -------
    freqs_pos : numpy array, shape (n_freqs)
        Array with positive frequencies.
    ps_pos : numpy array, shape (n_freqs)
        Spectrum for positive frequencies.
    """
    # Fourier transform data
    freqs = np.fft.fftfreq(x.size, 1. / freq)
    ps = np.abs(np.fft.fft(x))**2

    freqs_pos = freqs[freqs > 0]
    ps_pos = ps[freqs > 0]

    return freqs_pos, ps_pos


def roots(xch, freq):
    """
    Find number and positions of zero crossings, minimum and maximum values between crossings,
    standard deviation of minimum and maximum values between crossings,
    slope of minimum/maximum values (increase or decrease of amplitude)

    Uses solution from https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python

    Parameters
    ----------
    xch : numpy array, shape (n_times)
        Chunk for channel.
    freq : int, float
        Sampling frequency

    Returns
    -------
    n_roots : int
        Number of zero crossings in x
    std_min : float
        Standard deviations of minimal values of x between roots
    std_max : float
        Standard deviations of maximal values of x between roots
    slope_min: float
        Slope of minimal values of x between roots
    slope_max: float
        Slope of maximal values of x between roots
    """
    n_roots = 0
    std_min = 0
    std_max = 0
    slope_min = 0
    slope_max = 0

    zero_crossings = np.where(np.diff(xch > 0))[0]
    n = zero_crossings.shape[0]
    n_roots = n
    if n > 1:
        n_mins = n // 2 if (xch[0] > 0) else ((n // 2) - 1)
        mins = np.zeros(n_mins)
        mins_pos = np.zeros_like(mins)
        n_maxs = n // 2 if (xch[0] <= 0) else ((n // 2) - 1)
        maxs = np.zeros(n_maxs)
        maxs_pos = np.zeros_like(maxs)

        if xch[0] > 0:
            for j in range(n_mins):
                mins[j] = np.min(xch[zero_crossings[j * 2]:zero_crossings[j * 2 + 1]])
                mins_pos[j] = zero_crossings[j * 2] + np.argmin(
                    xch[zero_crossings[j * 2]:zero_crossings[j * 2 + 1]])
            for j in range(n_maxs):
                maxs[j] = np.max(xch[zero_crossings[j * 2 + 1]:zero_crossings[j * 2 + 2]])
                maxs_pos[j] = zero_crossings[j * 2 + 1] + np.argmax(
                    xch[zero_crossings[j * 2 + 1]:zero_crossings[j * 2 + 2]])
        else:
            for j in range(n_mins):
                mins[j] = np.min(xch[zero_crossings[j * 2 + 1]:zero_crossings[j * 2 + 2]])
                mins_pos[j] = zero_crossings[j * 2 + 1] + np.argmin(
                    xch[zero_crossings[j * 2 + 1]:zero_crossings[j * 2 + 2]])
            for j in range(n_maxs):
                maxs[j] = np.max(xch[zero_crossings[j * 2]:zero_crossings[j * 2 + 1]])
                maxs_pos[j] = zero_crossings[j * 2] + np.argmax(
                    xch[zero_crossings[j * 2]:zero_crossings[j * 2 + 1]])

        std_min = np.std(mins)
        std_max = np.std(maxs)

        t = (mins_pos - mins_pos.mean()) / freq
        y = mins - mins.mean()
        slope_min = (t.dot(y)) / (t.dot(t))

        t = (maxs_pos - maxs_pos.mean()) / freq
        y = maxs - maxs.mean()
        slope_max = (t.dot(y)) / (t.dot(t))

    return n_roots, std_min, std_max, slope_min, slope_max


def get_metrics_for_chunk(chunk, ch_names, freq, selected_features=None, selected_cross_features=None,
                          reference_channel='CZ', return_data=False):
    """
    Process chunk of data to metrics (both single-channel and cross-channel)

    Parameters
    ----------
    chunk : ndarray
        EEG data in array of shape (n_channels, n_samples).
    ch_names : list
        List of channel names in the order they are recorded and presented in chunk variable.
    freq : float
        Sampling frequency.
    selected_features : list
        List of features that are calculated from single-channel data.
    selected_cross_features : list
        List of features that are calculated from data of all channels.
    reference_channel : str
        Name of reference channel.

    Returns
    -------
    chunk_metrics : ndarray
        Array with calculated metrics.
    header : list
        List contanining names of calculated metrics.
    """
    if not selected_features:
        selected_features = ['mean',
                             'sd',
                             'skewness',
                             'kurtosis',
                             'min',
                             'max',
                             'median_abs',
                             'amp_entropy',
                             'psd_entropy',
                             'theta_power',
                             'alpha_power',
                             'mu_power',
                             'beta_power',
                             'dominant_freq',
                             'a2t_power',
                             'm2t_power',
                             'b2t_power',
                             'm2a_power',
                             'b2a_power',
                             'b2m_power']
    if not selected_cross_features:
        selected_cross_features = ['mean_diff',
                                   'min_diff',
                                   'max_diff',
                                   'mean_ratio',
                                   'min_ratio',
                                   'max_ratio']
    chunk_metrics = []
    header = []
    # sort channels, remove reference, scale to uV
    ch_names = [ch_name.upper() for ch_name in ch_names if not ('IMP' in ch_name.upper())]
    if not (reference_channel is None):
        ref_idx = ch_names.index(reference_channel.upper())
        chunk -= chunk[ref_idx]
    order = [x for _, x in sorted(zip(ch_names, range(len(ch_names)))) if not (_ == reference_channel)]
    ch_names = [ch_names[i] for i in order]
    # if np.log10(np.mean(np.abs(chunk[0, :]))) < 6:
    #     chunk = 1e6 * chunk[order]
    # else:
    #     chunk = chunk[order]
    chunk = chunk[order]
    # iterate through channels
    for i, (data, ch_name) in enumerate(zip(chunk, ch_names)):
        if ch_name == 'CZ':
            continue
        channel_metrics = calc_metrics(data, freq, selected_metrics=selected_features)
        chunk_metrics += [val for key, val in channel_metrics.items()]
        header += [(ch_name + '_' + key) for key, val in channel_metrics.items()]
    # calculate cross-channel metrics
    cross_metrics = calc_cross_metrics(chunk, ch_names, freq, selected_metrics=selected_cross_features)
    chunk_metrics += [val for key, val in cross_metrics.items()]
    header += [key for key, val in cross_metrics.items()]
    chunk_metrics = np.array(chunk_metrics)
    if not return_data:
        return chunk_metrics, header
    else:
        return chunk_metrics, header, chunk, ch_names
