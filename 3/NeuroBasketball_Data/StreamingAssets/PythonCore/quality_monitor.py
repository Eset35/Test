import pickle
import json
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from eeg_features import get_metrics_for_chunk, calc_spectrum

"""
Module that process EEG data and estimate the quality of signal.
It uses the fitted SVM model from sklearn on scaled data.
Classifier and scaler are loaded from pickle files.
"""


class QualityMonitor:
    def __init__(self, scaler, classifier, labels, extraction_settings):
        self._version = '2024.02.28.1'
        self.sfreq = 250
        self.buffer_len_windows = 120
        self.n_channels = 5
        self.ch_names = ['FZ', 'C3', 'CZ', 'C4', 'PZ']
        self.low_mean_amp_threshold = 100e-9  # V
        self.high_mean_amp_threshold = 250e-6  # V
        self.n_points_amp_check = 25
        self.allowed_decay_factor = 100
        self.peak_to_peak_threshold = 500e-6  # V
        self.freq_bands_split_frequency = 10  # Hz
        self.freq_bands_ratio_threshold = 75
        self.allowed_correlation = 0.8

        self.delay_before_lowering_threholds = 180  # s

        # Load pre-trained model from pickle
        # Model could be generated for example with fit_svm_model.py
        if type(scaler) == str:
            with open(scaler, 'rb') as f:
                self.scaler = pickle.load(f)
        elif type(scaler) == StandardScaler:
            self.scaler = scaler
        else:
            raise ValueError('Send path as string or StandardScaler object.')

        if type(classifier) == str:
            with open(classifier, 'rb') as f:
                self.classifier = pickle.load(f)
        elif type(classifier) == svm.SVC:
            self.classifier = classifier
        else:
            raise ValueError('Send path as string or svm.SVC object.')

        # Load labels
        with open(labels, 'r') as f:
            self.dict_of_labels = json.load(f)
        self.ha_labels = {
            'ha_0_0_0_0': -2,
            'ha_0_0_0_1': -3,
            'ha_0_0_1_0': -4,
            'ha_0_0_1_1': -5,
            'ha_0_1_0_0': -6,
            'ha_0_1_0_1': -7,
            'ha_0_1_1_0': -8,
            'ha_0_1_1_1': -9,
            'ha_1_0_0_0': -10,
            'ha_1_0_0_1': -11,
            'ha_1_0_1_0': -12,
            'ha_1_0_1_1': -13,
            'ha_1_1_0_0': -14,
            'ha_1_1_0_1': -15,
            'ha_1_1_1_0': -16
        }
        self.correlation_labels = {
            'c_1_1_0_0': -17,
            'c_1_0_1_0': -18,
            'c_1_0_0_1': -19,
            'c_0_1_1_0': -20,
            'c_0_1_0_1': -21,
            'c_0_0_1_1': -22,
            'c_1_1_1_0': -23,
            'c_1_1_0_1': -24,
            'c_1_0_1_1': -25,
            'c_0_1_1_1': -26,
            'c_1_1_1_1': -27
        }
        self.dict_of_labels = {**self.dict_of_labels, **self.ha_labels, **self.correlation_labels}
        self.labels_expansion = {
            'a': 'Alpha wave',
            'n': 'Noise',
            'g': 'Idling',
            'b': 'Blinking',
            's': 'Squinting',
            'j': 'Jaw clinch',
            'bm': 'Brow movement',
            'jl': 'Jaw movement, left',
            'jr': 'Jaw movement, right',
            'el': 'Eyes movement, left',
            'er': 'Eyes movement, right',
            'eu': 'Eyes movement, up',
            'ed': 'Eyes movement, down',
            'hl': 'Head movement, left',
            'hr': 'Head movement, right',
            'hu': 'Head movement, up',
            'hd': 'Head movement, down',
            'nd': 'Not defined',
            'ha_0_0_0_0': 'High amplitude in FZ, C3, C4, PZ',
            'ha_0_0_0_1': 'High amplitude in FZ, C3, C4',
            'ha_0_0_1_0': 'High amplitude in FZ, C3, PZ',
            'ha_0_0_1_1': 'High amplitude in FZ, C3',
            'ha_0_1_0_0': 'High amplitude in FZ, C4, PZ',
            'ha_0_1_0_1': 'High amplitude in FZ, C4',
            'ha_0_1_1_0': 'High amplitude in FZ, PZ',
            'ha_0_1_1_1': 'High amplitude in FZ',
            'ha_1_0_0_0': 'High amplitude in C3, C4, PZ',
            'ha_1_0_0_1': 'High amplitude in C3, C4',
            'ha_1_0_1_0': 'High amplitude in C3, PZ',
            'ha_1_0_1_1': 'High amplitude in C3',
            'ha_1_1_0_0': 'High amplitude in C4, PZ',
            'ha_1_1_0_1': 'High amplitude in C4',
            'ha_1_1_1_0': 'High amplitude in PZ',
            'c_1_1_0_0': 'High correlation in FZ-C3',
            'c_1_0_1_0': 'High correlation in FZ-C4',
            'c_1_0_0_1': 'High correlation in FZ-PZ',
            'c_0_1_1_0': 'High correlation in C3-C4',
            'c_0_1_0_1': 'High correlation in C3-PZ',
            'c_0_0_1_1': 'High correlation in C4-PZ',
            'c_1_1_1_0': 'High correlation in FZ-C3-C4',
            'c_1_1_0_1': 'High correlation in FZ-C3-PZ',
            'c_1_0_1_1': 'High correlation in FZ-C4-PZ',
            'c_0_1_1_1': 'High correlation in C3-C4-PZ',
            'c_1_1_1_1': 'High correlation in FZ-C3-C4-PZ'
        }

        self.baseline_conditions_thresholds = {
            'ha_duration_limit': 30,  # seconds
            'corr_duration_limit': 30,  # seconds
            'jaw_duration_limit': 30,  # seconds
            'no_blinks_duration_limit': 60,  # seconds
            'blinks_duration_limit': 60,  # seconds
            'blinks_fraction_limit': 40,  # percent
            'blinks_eyes_closed_limit': 6,

            'ha_fraction_limit': 25,  # percent
            'corr_fraction_limit': 25,  # percent
            'jaw_fraction_limit': 25,  # percent
            'blinks_lower_limit': 5,  # percent
            'blinks_upper_limit': 70,  # percent
            'blinks_eyes_closed_total_limit': 5  # percent
        }

        self.game_conditions_thresholds = {
            'ha_duration_limit': 60,  # seconds
            'ha_fraction_limit': 80,  # percent
            'la_duration_limit': 10,  # seconds
            'la_fraction_limit': 80,  # percent
            'corr_duration_limit': 60,  # seconds
            'corr_fraction_limit': 80,  # percent
            'jaw_duration_limit': 60,  # seconds
            'jaw_fraction_limit': 80,  # percent
            'no_blinks_duration_limit': 120,  # seconds
            'blinks_duration_limit': 120,  # seconds
            'blinks_fraction_limit': 40  # percent
        }

        # load settings from json
        with open(extraction_settings, 'r') as f:
            self.extraction_settings = json.load(f)
        self.win_length_samples = int(self.extraction_settings['win_length'] * self.sfreq * self.buffer_len_windows)

        # initialize buffers
        self.buffer = np.random.rand(self.n_channels, self.win_length_samples)
        self.quality_predictions = -np.ones(shape=self.buffer_len_windows, dtype=int)
        self.baseline_windows_rejection = np.zeros(shape=self.buffer_len_windows, dtype=int)
        self.last_state = -1
        self.samples_received = 0
        self.windows_predicted = 0

        # initialize quality scores
        self.quality = np.zeros(4, dtype=int)
        self._mean_abs_amp = np.zeros(4)
        self._mean_abs_amp_decay_factor = np.zeros(4)
        self._pp_amp = np.zeros(4)
        self._freq_bands_ratio = np.zeros(4)
        self._correlation_matrix = np.zeros((4, 4))

        # initialize quality flags
        self.ch_amplitude_fail = np.zeros(shape=4, dtype=bool)
        self.corr_flags = np.zeros(shape=4, dtype=bool)

    def version(self):
        return self._version

    def clean_buffer(self):
        # initialize buffers
        self.buffer = np.random.rand(self.n_channels, self.win_length_samples)
        self.quality_predictions = -np.ones(shape=self.buffer_len_windows, dtype=int)
        self.baseline_windows_rejection = np.zeros(shape=self.buffer_len_windows, dtype=int)
        self.last_state = -1
        self.samples_received = 0
        self.windows_predicted = 0

        # initialize quality scores
        self.quality = np.zeros(4, dtype=int)
        self._mean_abs_amp = np.zeros(4)
        self._mean_abs_amp_decay_factor = np.zeros(4)
        self._pp_amp = np.zeros(4)
        self._freq_bands_ratio = np.zeros(4)
        self._correlation_matrix = np.zeros((4, 4))

        # initialize quality flags
        self.ch_amplitude_fail = np.zeros(shape=4, dtype=bool)
        self.corr_flags = np.zeros(shape=4, dtype=bool)

    def lower_thresholds(self):
        self.high_mean_amp_threshold = 500e-6  # V
        self.peak_to_peak_threshold = 1000e-6  # V
        self.allowed_correlation = 0.9

    def log(self):
        log_string = f'QM version: {self._version}, '
        log_string += f'Last state label: {self.get_label(self.last_state)}, '
        log_string += f'Last state: {self.expand_label(self.get_label(self.last_state))}'
        return log_string

    def update_buffer(self, samples=None):
        if len(samples) >= (self.n_channels * self.win_length_samples):
            samples = samples[-self.n_channels * self.win_length_samples:]

        if len(samples) > 0:
            n_samples = len(samples)
            n_smpls_cut = n_samples // self.n_channels
            self.samples_received += n_smpls_cut

            # change thresholds if user fails to set cap for too long
            if self.samples_received >= self.sfreq * self.delay_before_lowering_threholds:
                self.lower_thresholds()

            # receive samples of data and append them to the end of buffer
            if (n_samples % self.n_channels) != 0:
                print('WARNING: Number of samples is not multiply of 5. Data loss may happen.')
                self.buffer[:, :self.buffer.shape[1] - n_smpls_cut] = self.buffer[:,
                                                                      n_smpls_cut:self.buffer.shape[1]]
                self.buffer[:, self.buffer.shape[1] - n_smpls_cut:] = \
                    np.reshape(samples[:n_smpls_cut * self.n_channels], (self.n_channels, n_smpls_cut), order='F')
            else:
                self.buffer[:, :self.buffer.shape[1] - n_smpls_cut] = self.buffer[:,
                                                                      n_smpls_cut:self.buffer.shape[1]]
                self.buffer[:, self.buffer.shape[1] - n_smpls_cut:] = \
                    np.reshape(samples, (self.n_channels, n_smpls_cut), order='F')

    def predict(self, rng=(-251, -1), update_state=True):
        # initialize amplitude, correlation and frequency ratio threshold variables
        mean_abs_amplitudes_fail = np.zeros(shape=4, dtype=bool)
        peak_to_peak_amplitudes_fail = np.zeros(shape=4, dtype=bool)
        freq_bands_ratio_fail = np.zeros(shape=4, dtype=bool)

        classifier_prediction = -1
        idx0 = rng[0]
        idx1 = rng[1]

        # per channel check of the violation of:
        # 1. mean absolute amplitude threshold
        # 2. peak to peak amplitude threshold
        # 3. frequency bands ratio threshold
        # 4. correlation threshold
        for i in range(4):
            if i < 2:
                chunk_data = self.buffer[i, idx0:idx1]
            elif i >= 2:
                chunk_data = self.buffer[i + 1, idx0:idx1]

            freqs, ps = calc_spectrum(chunk_data, self.sfreq)

            # calculate values of amplitude, correlation and frequency bands ratio
            self._mean_abs_amp[i] = np.mean(np.abs(chunk_data))
            self._mean_abs_amp_decay_factor[i] = (np.mean(np.abs(chunk_data[:self.n_points_amp_check])) /
                                                  np.mean(np.abs(chunk_data[-self.n_points_amp_check:])))
            self._pp_amp[i] = np.max(chunk_data) - np.min(chunk_data)
            self._freq_bands_ratio[i] = (np.sum(ps[freqs > self.freq_bands_split_frequency]) /
                                         np.sum(ps[freqs <= self.freq_bands_split_frequency]))

            # mean absolute amplitude
            if (self._mean_abs_amp[i] <= self.low_mean_amp_threshold) or \
                    (self._mean_abs_amp[i] >= self.high_mean_amp_threshold):
                mean_abs_amplitudes_fail[i] = True
            else:
                # check mean absolute amplitude decay factor
                if self._mean_abs_amp_decay_factor[i] >= self.allowed_decay_factor:
                    mean_abs_amplitudes_fail[i] = True
                else:
                    mean_abs_amplitudes_fail[i] = False

            # peak to peak amplitude
            if self._pp_amp[i] >= self.peak_to_peak_threshold:
                peak_to_peak_amplitudes_fail[i] = True
            else:
                peak_to_peak_amplitudes_fail[i] = False

            # frequency bands ratio
            if self._freq_bands_ratio[i] >= self.freq_bands_ratio_threshold:
                freq_bands_ratio_fail[i] = True
            else:
                freq_bands_ratio_fail[i] = False

        # correlations
        correlation_fail = False
        self.corr_flags = np.zeros(shape=4, dtype=bool)
        self._correlation_matrix = np.corrcoef(self.buffer[np.array([0, 1, 3, 4]), idx0:idx1]) ** 2

        for cidx1 in range(3):
            if cidx1 >= 2:
                buf_idx1 = cidx1 + 1
            else:
                buf_idx1 = cidx1

            for cidx2 in range(cidx1 + 1, 4, 1):
                if cidx2 >= 2:
                    buf_idx2 = cidx2 + 1
                else:
                    buf_idx2 = cidx2

                if ((self._correlation_matrix[cidx1, cidx2] >= self.allowed_correlation) and
                        (np.mean(np.abs(self.buffer[buf_idx1, idx0:idx1])) >= self.low_mean_amp_threshold) and
                        (np.mean(np.abs(self.buffer[buf_idx2, idx0:idx1])) >= self.low_mean_amp_threshold)):
                    correlation_fail = True
                    self.corr_flags[cidx1] = True
                    self.corr_flags[cidx2] = True

        # check parameters of amplitudes and assign label
        for i in range(4):
            if mean_abs_amplitudes_fail[i] or peak_to_peak_amplitudes_fail[i] or freq_bands_ratio_fail[i]:
                self.ch_amplitude_fail[i] = True
            else:
                self.ch_amplitude_fail[i] = False

        if (self.ch_amplitude_fail[0] or self.ch_amplitude_fail[1] or
                self.ch_amplitude_fail[2] or self.ch_amplitude_fail[3]):
            classifier_prediction = self.dict_of_labels[f'ha_{int(not self.ch_amplitude_fail[0])}' +
                                                        f'_{int(not self.ch_amplitude_fail[1])}' +
                                                        f'_{int(not self.ch_amplitude_fail[2])}' +
                                                        f'_{int(not self.ch_amplitude_fail[3])}']

        # check parameters of correlation and assign label
        elif correlation_fail:
            classifier_prediction = self.dict_of_labels[f'c_{int(self.corr_flags[0])}' +
                                                        f'_{int(self.corr_flags[1])}' +
                                                        f'_{int(self.corr_flags[2])}' +
                                                        f'_{int(self.corr_flags[3])}']

        else:
            # predict state for last chunk
            data = self.buffer[:, idx0:idx1]

            try:
                chunk_metrics, metrics_header = \
                    get_metrics_for_chunk(data, self.ch_names, self.sfreq,
                                          selected_features=self.extraction_settings['selected_features'],
                                          selected_cross_features=self.extraction_settings[
                                              'selected_cross_features'],
                                          reference_channel='CZ',
                                          return_data=False)

                classifier_prediction = \
                    self.classifier.predict(self.scaler.transform(chunk_metrics.reshape(1, -1)))[0]
            except:
                classifier_prediction = -1

        if update_state:
            self.last_state = classifier_prediction

            # update quality
            self.update_quality()

        else:
            return classifier_prediction

    def update_quality(self):
        # EEG quality codes:
        # 0 = bad quality, color in red
        # 1 = good quality, color in green
        # 2 = biological artifact, color in blue

        last_state_label = self.get_label(self.last_state)
        # set quality to 1 if last prediction is good
        if last_state_label == 'g':
            self.quality = np.ones_like(self.quality, dtype=int)
        elif last_state_label in ['b', 's', 'bm']:
            self.quality[0] = 2
            self.quality[1:] = np.ones_like(self.quality[1:], dtype=int)
        elif last_state_label == 'j':
            self.quality[0] = 1
            self.quality[1] = 2
            self.quality[2] = 2
            self.quality[3] = 1
        elif last_state_label == 'jl':
            self.quality[0] = 1
            self.quality[1] = 2
            self.quality[2] = 1
            self.quality[3] = 1
        elif last_state_label == 'jr':
            self.quality[0] = 1
            self.quality[1] = 1
            self.quality[2] = 2
            self.quality[3] = 1
        elif last_state_label == 'a':
            self.quality[0] = 1
            self.quality[1] = 1
            self.quality[2] = 1
            self.quality[3] = 2

        # set quality to 0 to particular channels according to labels
        # if last prediction is not good
        elif 'ha' in last_state_label:
            self.quality = np.array([int(x) for x in last_state_label[3:].split('_')], dtype=int)

            # check correlations for high amplitude containing chunks
            for i in range(4):
                if self.quality[i] == 1:
                    self.quality[i] = int(not bool(int(self.corr_flags[i])))

        elif 'c' in last_state_label:
            self.quality = np.array([int(not bool(int(x))) for x in last_state_label[2:].split('_')], dtype=int)

        else:
            self.quality = np.ones_like(self.quality, dtype=int)

    def get_colors(self):
        colors = dict()
        for i, ch in zip(range(4), ['FZ', 'C3', 'C4', 'PZ']):
            if self.quality[i] == 0:
                colors[ch] = [0xd6, 0x49, 0x33]
            elif self.quality[i] == 1:
                colors[ch] = [0x98, 0xbf, 0x64]
            elif self.quality[i] == 2:
                colors[ch] = [0x46, 0x82, 0xb4]
        return json.dumps(colors)

    def get_label(self, prediction):
        label_keys = [k for k, v in self.dict_of_labels.items()]
        label_vals = [v for k, v in self.dict_of_labels.items()]
        if prediction == -1:
            return 'nd'
        return label_keys[label_vals.index(prediction)]

    def expand_label(self, label):
        return self.labels_expansion[label]

    def baseline_status_online(self, eyes='opened'):
        # Initialize default response
        response = {
            'amp_artifacts': False,
            'ha_FZ': False,
            'la_FZ': False,
            'ha_C3': False,
            'la_C3': False,
            'ha_C4': False,
            'la_C4': False,
            'ha_PZ': False,
            'la_PZ': False,
            'corr_artifacts': False,
            'corr_FZ': False,
            'corr_C3': False,
            'corr_C4': False,
            'corr_PZ': False,
            'jaw_artifacts': False,
            'no_blinks': False,
            'too_many_blinks': False,
            'blinks_in_eyes_closed_bl': False
        }

        # Update received data and predictions buffer
        win_received = self.samples_received // self.sfreq
        if self.samples_received <= self.win_length_samples:
            # Buffer not overfilled
            for i in range(self.windows_predicted, win_received):
                # indexes for received windows
                origin = self.win_length_samples - self.samples_received
                idx1 = origin + i * self.sfreq
                idx2 = origin + (i + 1) * self.sfreq
                self.quality_predictions[i] = self.predict(rng=(idx1, idx2), update_state=False)
            self.windows_predicted = win_received

        else:
            # Buffer overfilled, data may be shifted
            for i in range(self.buffer_len_windows):
                self.quality_predictions[i] = self.predict(rng=(i * self.sfreq, (i + 1) * self.sfreq),
                                                           update_state=False)
            self.windows_predicted = self.buffer_len_windows

        # Evaluate conditions for online baseline QM
        online_predictions = self.quality_predictions[:win_received]

        # High/low amplitude detected for more than_{ha_duration_limit}_in channels_FZ_C3_C4_PZ
        n_win_to_evaluate = int(self.baseline_conditions_thresholds['ha_duration_limit'] //
                                self.extraction_settings['win_length'])
        if win_received >= n_win_to_evaluate:
            fail = True
            for prediction in online_predictions[-n_win_to_evaluate:]:
                if 'ha' not in self.get_label(prediction):
                    fail = False
                    break
            if fail:
                response['amp_artifacts'] = True

                # find which electrodes cause problem in last chunk
                for i, ch in enumerate(['FZ', 'C3', 'C4', 'PZ']):
                    if self.ch_amplitude_fail[i]:
                        if self._mean_abs_amp[i] >= self.high_mean_amp_threshold:
                            response['ha_' + ch] = True
                        elif self._mean_abs_amp[i] <= self.low_mean_amp_threshold:
                            response['la_' + ch] = True

                return json.dumps(response)

        # High correlation detected for more than_{corr_duration_limit}_between channels_FZ_C3_C4_PZ
        n_win_to_evaluate = int(self.baseline_conditions_thresholds['corr_duration_limit'] //
                                self.extraction_settings['win_length'])
        if win_received >= n_win_to_evaluate:
            fail = True
            for prediction in online_predictions[-n_win_to_evaluate:]:
                if 'c' not in self.get_label(prediction):
                    fail = False
                    break
            if fail:
                response['corr_artifacts'] = True

                # find which electrodes cause problem in last chunk
                for i, ch in enumerate(['FZ', 'C3', 'C4', 'PZ']):
                    if self.corr_flags[i]:
                        response['corr_' + ch] = True

                return json.dumps(response)

        # Jaw muscle artifacts detected for more than_{jaw_duration_limit}
        n_win_to_evaluate = int(self.baseline_conditions_thresholds['jaw_duration_limit'] //
                                self.extraction_settings['win_length'])
        if win_received >= n_win_to_evaluate:
            fail = True
            for prediction in online_predictions[-n_win_to_evaluate:]:
                if 'j' not in self.get_label(prediction):
                    fail = False
                    break
            if fail:
                response['jaw_artifacts'] = True
                return json.dumps(response)

        if eyes == 'opened':

            # No blinks detected in past_{no_blinks_duration_limit}
            n_win_to_evaluate = int(self.baseline_conditions_thresholds['no_blinks_duration_limit'] //
                                    self.extraction_settings['win_length'])
            if win_received >= n_win_to_evaluate:
                fail = True
                for prediction in online_predictions[-n_win_to_evaluate:]:
                    if 'b' in self.get_label(prediction):
                        fail = False
                        break
                if fail:
                    response['no_blinks'] = True
                    return json.dumps(response)

            # Blinks detected in more than_{blinks_duration_limit}
            n_win_to_evaluate = int(self.baseline_conditions_thresholds['blinks_duration_limit'] //
                                    self.extraction_settings['win_length'])
            if win_received >= n_win_to_evaluate:
                fail = False
                n_blinks = 0
                for prediction in online_predictions[-n_win_to_evaluate:]:
                    if 'b' in self.get_label(prediction):
                        n_blinks += 1
                if (100 * n_blinks / n_win_to_evaluate) >= self.baseline_conditions_thresholds['blinks_fraction_limit']:
                    fail = True
                if fail:
                    response['too_many_blinks'] = True
                    return json.dumps(response)

        elif eyes == 'closed':

            # More than_{blinks_eyes_closed_limit}_blinks detected (for EYES CLOSED)
            fail = False
            n = 0
            for prediction in online_predictions:
                if 'b' in self.get_label(prediction):
                    n += 1
                if n >= self.baseline_conditions_thresholds['blinks_eyes_closed_limit']:
                    fail = True
                    break
            if fail:
                response['blinks_in_eyes_closed_bl'] = True
                return json.dumps(response)

        return json.dumps(response)

    def baseline_quality(self, eyes='opened'):
        for i in range(self.buffer_len_windows):
            self.quality_predictions[i] = self.predict(rng=(i * self.sfreq, (i + 1) * self.sfreq),
                                                       update_state=False)

        # Initialize default response
        response = {
            'amp_artifacts_high_fraction': False,
            'corr_artifacts_high_fraction': False,
            'jaw_artifacts_high_fraction': False,
            'low_blinks_fraction': False,
            'high_blinks_fraction': False,
            'blinks_eyes_closed_high_fraction': False,
            'event_counts': {
                'n_idle': 0,
                'n_alpha': 0,
                'n_blink': 0,
                'n_jaw': 0,
                'n_jaw_left': 0,
                'n_jaw_right': 0,
                'n_ha': 0,
                'n_ha_fz': 0,
                'n_ha_c3': 0,
                'n_ha_c4': 0,
                'n_ha_pz': 0,
                'n_corr': 0,
                'n_corr_fz_c3': 0,
                'n_corr_fz_c4': 0,
                'n_corr_fz_pz': 0,
                'n_corr_c3_c4': 0,
                'n_corr_c3_pz': 0,
                'n_corr_c4_pz': 0,
                'n_corr_fz_c3_c4': 0,
                'n_corr_fz_c3_pz': 0,
                'n_corr_fz_c4_pz': 0,
                'n_corr_c3_c4_pz': 0,
                'n_corr_fz_c3_c4_pz': 0
            },
            'windows_for_rejection': [False] * self.buffer_len_windows,
            'quality_labels': [self.get_label(prediction) for prediction in self.quality_predictions],
            'windows_for_rejection_by_channel': [[False] * 4] * self.buffer_len_windows
        }

        n_idle = 0
        n_alpha = 0
        n_blink = 0
        n_jaw = 0
        n_jaw_left = 0
        n_jaw_right = 0
        n_ha = 0
        n_ha_fz = 0
        n_ha_c3 = 0
        n_ha_c4 = 0
        n_ha_pz = 0
        n_corr = 0
        n_corr_fz_c3 = 0
        n_corr_fz_c4 = 0
        n_corr_fz_pz = 0
        n_corr_c3_c4 = 0
        n_corr_c3_pz = 0
        n_corr_c4_pz = 0
        n_corr_fz_c3_c4 = 0
        n_corr_fz_c3_pz = 0
        n_corr_fz_c4_pz = 0
        n_corr_c3_c4_pz = 0
        n_corr_fz_c3_c4_pz = 0

        for i, prediction in enumerate(self.quality_predictions):
            short_label = self.get_label(prediction)
            if short_label == 'g':
                n_idle += 1
            if short_label == 'a':
                n_alpha += 1
            if short_label == 'b':
                n_blink += 1
            if short_label[:1] == 'j':
                n_jaw += 1
            if short_label == 'jl':
                n_jaw_left += 1
            if short_label == 'jr':
                n_jaw_right += 1
            if short_label[:2] == 'ha':
                n_ha += 1
            if (short_label[:2] == 'ha') and (short_label.split('_')[1] == '0'):
                n_ha_fz += 1
            if (short_label[:2] == 'ha') and (short_label.split('_')[2] == '0'):
                n_ha_c3 += 1
            if (short_label[:2] == 'ha') and (short_label.split('_')[3] == '0'):
                n_ha_c4 += 1
            if (short_label[:2] == 'ha') and (short_label.split('_')[4] == '0'):
                n_ha_pz += 1
            if short_label[:1] == 'c':
                n_corr += 1
            if (short_label[:1] == 'c') and (short_label.split('_')[1] == '1') and (short_label.split('_')[2] == '1'):
                n_corr_fz_c3 += 1
            if (short_label[:1] == 'c') and (short_label.split('_')[1] == '1') and (short_label.split('_')[3] == '1'):
                n_corr_fz_c4 += 1
            if (short_label[:1] == 'c') and (short_label.split('_')[1] == '1') and (short_label.split('_')[4] == '1'):
                n_corr_fz_pz += 1
            if (short_label[:1] == 'c') and (short_label.split('_')[2] == '1') and (short_label.split('_')[3] == '1'):
                n_corr_c3_c4 += 1
            if (short_label[:1] == 'c') and (short_label.split('_')[2] == '1') and (short_label.split('_')[4] == '1'):
                n_corr_c3_pz += 1
            if (short_label[:1] == 'c') and (short_label.split('_')[3] == '1') and (short_label.split('_')[4] == '1'):
                n_corr_c4_pz += 1
            if ((short_label[:1] == 'c') and (short_label.split('_')[1] == '1') and
                    (short_label.split('_')[2] == '1') and (short_label.split('_')[3] == '1')):
                n_corr_fz_c3_c4 += 1
            if ((short_label[:1] == 'c') and (short_label.split('_')[1] == '1') and
                    (short_label.split('_')[2] == '1') and (short_label.split('_')[4] == '1')):
                n_corr_fz_c3_pz += 1
            if ((short_label[:1] == 'c') and (short_label.split('_')[1] == '1') and
                    (short_label.split('_')[3] == '1') and (short_label.split('_')[4] == '1')):
                n_corr_fz_c4_pz += 1
            if ((short_label[:1] == 'c') and (short_label.split('_')[1] == '2') and
                    (short_label.split('_')[3] == '1') and (short_label.split('_')[4] == '1')):
                n_corr_c3_c4_pz += 1
            if ((short_label[:1] == 'c') and (short_label.split('_')[1] == '2') and
                    (short_label.split('_')[2] == '1') and (short_label.split('_')[3] == '1') and
                    (short_label.split('_')[4] == '1')):
                n_corr_fz_c3_c4_pz += 1

            # Mark windows for rejection
            if (short_label[:2] == 'ha') or (short_label[:1] == 'c') or (short_label[:1] == 'j'):
                response['windows_for_rejection'][i] = True
            windows_for_rejection_by_channel = [False] * 4
            for ch_idx in range(4):
                if (short_label[:2] == 'ha') and (short_label.split('_')[ch_idx + 1] == '0'):
                    windows_for_rejection_by_channel[ch_idx] = True
                if (short_label[:1] == 'c') and (short_label.split('_')[ch_idx + 1] == '1'):
                    windows_for_rejection_by_channel[ch_idx] = True
            if short_label == 'j':
                windows_for_rejection_by_channel[1] = True
                windows_for_rejection_by_channel[2] = True
            if short_label == 'jl':
                windows_for_rejection_by_channel[1] = True
            if short_label == 'jr':
                windows_for_rejection_by_channel[2] = True

            response['windows_for_rejection_by_channel'][i] = windows_for_rejection_by_channel

        # Evaluate conditions for final baseline QM
        # Fraction of high/low amplitude windows more/less than_{ha_fraction_limit}
        if (100 * n_ha / self.buffer_len_windows) >= self.baseline_conditions_thresholds['ha_fraction_limit']:
            response['amp_artifacts_high_fraction'] = True

        # Fraction of high correlation windows more than_{corr_fraction_limit}
        if (100 * n_corr / self.buffer_len_windows) >= self.baseline_conditions_thresholds['corr_fraction_limit']:
            response['corr_artifacts_high_fraction'] = True

        # Fraction of jaw muscle artifacts windows more than_{jaw_fraction_limit}
        if (100 * n_jaw / self.buffer_len_windows) >= self.baseline_conditions_thresholds['jaw_fraction_limit']:
            response['jaw_artifacts_high_fraction'] = True

        if eyes == 'opened':

            # Fraction of windows with blinks less than_{blinks_lower_limit}
            if (100 * n_blink / self.buffer_len_windows) <= self.baseline_conditions_thresholds['blinks_lower_limit']:
                response['low_blinks_fraction'] = True

            # Fraction of windows with blinks more than_{blinks_upper_limit}
            if (100 * n_blink / self.buffer_len_windows) >= self.baseline_conditions_thresholds['blinks_upper_limit']:
                response['high_blinks_fraction'] = True

        elif eyes == 'closed':

            # Fraction of windows with blinks more than_{blinks_eyes_closed_total_limit}
            if (100 * n_blink / self.buffer_len_windows) >= self.baseline_conditions_thresholds['blinks_eyes_closed_total_limit']:
                response['blinks_eyes_closed_high_fraction'] = True

        # Put event counts to response
        response['event_counts']['n_idle'] = n_idle
        response['event_counts']['n_alpha'] = n_alpha
        response['event_counts']['n_blink'] = n_blink
        response['event_counts']['n_jaw'] = n_jaw
        response['event_counts']['n_jaw_left'] = n_jaw_left
        response['event_counts']['n_jaw_right'] = n_jaw_right
        response['event_counts']['n_ha'] = n_ha
        response['event_counts']['n_ha_fz'] = n_ha_fz
        response['event_counts']['n_ha_c3'] = n_ha_c3
        response['event_counts']['n_ha_c4'] = n_ha_c4
        response['event_counts']['n_ha_pz'] = n_ha_pz
        response['event_counts']['n_corr'] = n_corr
        response['event_counts']['n_corr_fz_c3'] = n_corr_fz_c3
        response['event_counts']['n_corr_fz_c4'] = n_corr_fz_c4
        response['event_counts']['n_corr_fz_pz'] = n_corr_fz_pz
        response['event_counts']['n_corr_c3_c4'] = n_corr_c3_c4
        response['event_counts']['n_corr_c3_pz'] = n_corr_c3_pz
        response['event_counts']['n_corr_c4_pz'] = n_corr_c4_pz
        response['event_counts']['n_corr_fz_c3_c4'] = n_corr_fz_c3_c4
        response['event_counts']['n_corr_fz_c3_pz'] = n_corr_fz_c3_pz
        response['event_counts']['n_corr_fz_c4_pz'] = n_corr_fz_c4_pz
        response['event_counts']['n_corr_c3_c4_pz'] = n_corr_c3_c4_pz
        response['event_counts']['n_corr_fz_c3_c4_pz'] = n_corr_fz_c3_c4_pz

        # Put marked windows for rejection to Class attribute
        self.baseline_windows_rejection = response['windows_for_rejection']

        return json.dumps(response)

    def game_status_online(self):
        # Initialize default response
        response = {
            'amp_artifacts': False,
            'ha_FZ': False,
            'la_FZ': False,
            'ha_C3': False,
            'la_C3': False,
            'ha_C4': False,
            'la_C4': False,
            'ha_PZ': False,
            'la_PZ': False,
            'corr_artifacts': False,
            'corr_FZ': False,
            'corr_C3': False,
            'corr_C4': False,
            'corr_PZ': False,
            'jaw_artifacts': False,
            'no_blinks': False,
            'too_many_blinks': False
        }

        # Update received data and predictions buffer
        win_received = self.samples_received // self.sfreq
        if self.samples_received <= self.win_length_samples:
            # Buffer not overfilled
            for i in range(self.windows_predicted, win_received):
                # indexes for received windows
                origin = self.win_length_samples - self.samples_received
                idx1 = origin + i * self.sfreq
                idx2 = origin + (i + 1) * self.sfreq
                self.quality_predictions[i] = self.predict(rng=(idx1, idx2), update_state=False)
            self.windows_predicted = win_received

        else:
            # Buffer overfilled, data may be shifted
            for i in range(self.buffer_len_windows):
                self.quality_predictions[i] = self.predict(rng=(i * self.sfreq, (i + 1) * self.sfreq),
                                                           update_state=False)
            self.windows_predicted = self.buffer_len_windows

        # Evaluate conditions for online baseline QM
        online_predictions = self.quality_predictions[:win_received]

        # Low amplitude detected for more than_{ha_duration_limit}_in channels_FZ_C3_C4_PZ
        n_win_to_evaluate = int(self.game_conditions_thresholds['la_duration_limit'] //
                                self.extraction_settings['win_length'])
        if win_received >= n_win_to_evaluate:
            n_detected_la = 0

            for prediction in online_predictions[-n_win_to_evaluate:]:
                if 'ha' in self.get_label(prediction):
                    n_detected_la += 1

            fraction_la = 100 * n_detected_la / n_win_to_evaluate

            if fraction_la >= self.game_conditions_thresholds['la_fraction_limit']:
                # find which electrodes cause problem in last chunk
                for i, ch in enumerate(['FZ', 'C3', 'C4', 'PZ']):
                    if self.ch_amplitude_fail[i]:
                        if self._mean_abs_amp[i] <= self.low_mean_amp_threshold:
                            response['amp_artifacts'] = True
                            response['la_' + ch] = True

            if response['la_FZ'] or response['la_C3'] or response['la_C4'] or response['la_PZ']:
                return json.dumps(response)

        # High amplitude detected for more than_{ha_duration_limit}_in channels_FZ_C3_C4_PZ
        n_win_to_evaluate = int(self.game_conditions_thresholds['ha_duration_limit'] //
                                self.extraction_settings['win_length'])
        if win_received >= n_win_to_evaluate:
            n_detected_ha = 0

            for prediction in online_predictions[-n_win_to_evaluate:]:
                if 'ha' in self.get_label(prediction):
                    n_detected_ha += 1

            fraction_ha = 100 * n_detected_ha / n_win_to_evaluate

            if fraction_ha >= self.game_conditions_thresholds['ha_fraction_limit']:
                # find which electrodes cause problem in last chunk
                for i, ch in enumerate(['FZ', 'C3', 'C4', 'PZ']):
                    if self.ch_amplitude_fail[i]:
                        if self._mean_abs_amp[i] >= self.high_mean_amp_threshold:
                            response['amp_artifacts'] = True
                            response['ha_' + ch] = True

                return json.dumps(response)

        # High correlation detected for more than_{corr_duration_limit}_between channels_FZ_C3_C4_PZ
        n_win_to_evaluate = int(self.game_conditions_thresholds['corr_duration_limit'] //
                                self.extraction_settings['win_length'])
        if win_received >= n_win_to_evaluate:
            n_detected_corr = 0

            for prediction in online_predictions[-n_win_to_evaluate:]:
                if 'c' in self.get_label(prediction):
                    n_detected_corr += 1

            fraction_corr = 100 * n_detected_corr / n_win_to_evaluate

            if fraction_corr >= self.game_conditions_thresholds['corr_fraction_limit']:
                response['corr_artifacts'] = True

                # find which electrodes cause problem in last chunk
                for i, ch in enumerate(['FZ', 'C3', 'C4', 'PZ']):
                    if self.corr_flags[i]:
                        response['corr_' + ch] = True

                return json.dumps(response)

        # Jaw muscle artifacts detected for more than_{jaw_duration_limit}
        n_win_to_evaluate = int(self.game_conditions_thresholds['jaw_duration_limit'] //
                                self.extraction_settings['win_length'])
        if win_received >= n_win_to_evaluate:
            n_detected_jaw = 0

            for prediction in online_predictions[-n_win_to_evaluate:]:
                if 'j' in self.get_label(prediction):
                    n_detected_jaw += 1

            fraction_jaw = 100 * n_detected_jaw / n_win_to_evaluate

            if fraction_jaw >= self.game_conditions_thresholds['jaw_fraction_limit']:
                response['jaw_artifacts'] = True
                return json.dumps(response)

        # No blinks detected in past_{no_blinks_duration_limit}
        n_win_to_evaluate = int(self.game_conditions_thresholds['no_blinks_duration_limit'] //
                                self.extraction_settings['win_length'])
        if win_received >= n_win_to_evaluate:
            fail = True
            for prediction in online_predictions[-n_win_to_evaluate:]:
                if 'b' in self.get_label(prediction):
                    fail = False
                    break
            if fail:
                response['no_blinks'] = True
                return json.dumps(response)

        # Blinks detected in more than_{blinks_duration_limit}
        n_win_to_evaluate = int(self.game_conditions_thresholds['blinks_duration_limit'] //
                                self.extraction_settings['win_length'])
        if win_received >= n_win_to_evaluate:
            n_blinks = 0
            for prediction in online_predictions[-n_win_to_evaluate:]:
                if 'b' in self.get_label(prediction):
                    n_blinks += 1
            fraction_blinks = 100 * n_blinks / n_win_to_evaluate
            if fraction_blinks >= self.game_conditions_thresholds['blinks_fraction_limit']:
                response['too_many_blinks'] = True
                return json.dumps(response)

        return json.dumps(response)
