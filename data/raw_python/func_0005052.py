def decode_events(self, encoded, transition=None, p_state=None, p_init=None):
        '''Decode labeled events into (time, value) pairs

        Real-valued inputs are thresholded at 0.5.

        Optionally, viterbi decoding can be applied to each event class.

        Parameters
        ----------
        encoded : np.ndarray, shape=(n_frames, m)
            Frame-level annotation encodings as produced by ``encode_events``.

        transition : None or np.ndarray [shape=(2, 2) or (m, 2, 2)]
            Optional transition matrix for each event, used for Viterbi

        p_state : None or np.ndarray [shape=(m,)]
            Optional marginal probability for each event

        p_init : None or np.ndarray [shape=(m,)]
            Optional marginal probability for each event

        Returns
        -------
        [(time, value)] : iterable of tuples
            where `time` is the event time and `value` is an
            np.ndarray, shape=(m,) of the encoded value at that time

        See Also
        --------
        librosa.sequence.viterbi_binary
        '''
        if np.isrealobj(encoded):
            if transition is None:
                encoded = (encoded >= 0.5)
            else:
                encoded = viterbi_binary(encoded.T, transition,
                                         p_state=p_state,
                                         p_init=p_init).T

        times = times_like(encoded,
                           sr=self.sr,
                           hop_length=self.hop_length,
                           axis=0)

        return zip(times, encoded)