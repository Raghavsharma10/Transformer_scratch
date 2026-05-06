def decode_intervals(self, encoded, duration=None, multi=True, sparse=False,
                         transition=None, p_state=None, p_init=None):
        '''Decode labeled intervals into (start, end, value) triples

        Parameters
        ----------
        encoded : np.ndarray, shape=(n_frames, m)
            Frame-level annotation encodings as produced by
            ``encode_intervals``

        duration : None or float > 0
            The max duration of the annotation (in seconds)
            Must be greater than the length of encoded array.

        multi : bool
            If true, allow multiple labels per input frame.
            If false, take the most likely label per input frame.

        sparse : bool
            If true, values are returned as indices, not one-hot.
            If false, values are returned as one-hot encodings.

            Only applies when `multi=False`.

        transition : None or np.ndarray [shape=(m, m) or (2, 2) or (m, 2, 2)]
            Optional transition matrix for each interval, used for Viterbi
            decoding.  If `multi=True`, then transition should be `(2, 2)` or
            `(m, 2, 2)`-shaped.  If `multi=False`, then transition should be
            `(m, m)`-shaped.

        p_state : None or np.ndarray [shape=(m,)]
            Optional marginal probability for each label.

        p_init : None or np.ndarray [shape=(m,)]
            Optional marginal probability for each label.

        Returns
        -------
        [(start, end, value)] : iterable of tuples
            where `start` and `end` are the interval boundaries (in seconds)
            and `value` is an np.ndarray, shape=(m,) of the encoded value
            for this interval.
        '''
        if np.isrealobj(encoded):
            if multi:
                if transition is None:
                    encoded = encoded >= 0.5
                else:
                    encoded = viterbi_binary(encoded.T, transition,
                                             p_init=p_init, p_state=p_state).T
            elif sparse and encoded.shape[1] > 1:
                # map to argmax if it's densely encoded (logits)
                if transition is None:
                    encoded = np.argmax(encoded, axis=1)[:, np.newaxis]
                else:
                    encoded = viterbi_discriminative(encoded.T, transition,
                                                     p_init=p_init,
                                                     p_state=p_state)[:, np.newaxis]
            elif not sparse:
                # if dense and multi, map to one-hot encoding
                if transition is None:
                    encoded = (encoded == np.max(encoded, axis=1, keepdims=True))
                else:
                    encoded_ = viterbi_discriminative(encoded.T, transition,
                                                      p_init=p_init,
                                                      p_state=p_state)
                    # Map to one-hot encoding
                    encoded = np.zeros(encoded.shape, dtype=bool)
                    encoded[np.arange(len(encoded_)), encoded_] = True

        if duration is None:
            # 1+ is fair here, because encode_intervals already pads
            duration = 1 + encoded.shape[0]
        else:
            duration = 1 + time_to_frames(duration,
                                          sr=self.sr,
                                          hop_length=self.hop_length)

        # [0, duration] inclusive
        times = times_like(duration + 1,
                           sr=self.sr, hop_length=self.hop_length)

        # Find the change-points of the rows
        if sparse:
            idx = np.where(encoded[1:] != encoded[:-1])[0]
        else:
            idx = np.where(np.max(encoded[1:] != encoded[:-1], axis=-1))[0]

        idx = np.unique(np.append(idx, encoded.shape[0]))
        delta = np.diff(np.append(-1, idx))

        # Starting positions can be integrated from changes
        position = np.cumsum(np.append(0, delta))

        return [(times[p], times[p + d], encoded[p])
                for (p, d) in zip(position, delta)]