def transform_annotation(self, ann, duration):
        '''Apply the chord transformation.

        Parameters
        ----------
        ann : jams.Annotation
            The chord annotation

        duration : number > 0
            The target duration

        Returns
        -------
        data : dict
            data['pitch'] : np.ndarray, shape=(n, 12)
            data['root'] : np.ndarray, shape=(n, 13) or (n, 1)
            data['bass'] : np.ndarray, shape=(n, 13) or (n, 1)

            `pitch` is a binary matrix indicating pitch class
            activation at each frame.

            `root` is a one-hot matrix indicating the chord
            root's pitch class at each frame.

            `bass` is a one-hot matrix indicating the chord
            bass (lowest note) pitch class at each frame.

            If sparsely encoded, `root` and `bass` are integers
            in the range [0, 12] where 12 indicates no chord.

            If densely encoded, `root` and `bass` have an extra
            final dimension which is active when there is no chord
            sounding.
        '''
        # Construct a blank annotation with mask = 0
        intervals, chords = ann.to_interval_values()

        # Get the dtype for root/bass
        if self.sparse:
            dtype = np.int
        else:
            dtype = np.bool

        # If we don't have any labeled intervals, fill in a no-chord
        if not chords:
            intervals = np.asarray([[0, duration]])
            chords = ['N']

        # Suppress all intervals not in the encoder
        pitches = []
        roots = []
        basses = []

        # default value when data is missing
        if self.sparse:
            fill = 12
        else:
            fill = False

        for chord in chords:
            # Encode the pitches
            root, semi, bass = mir_eval.chord.encode(chord)
            pitches.append(np.roll(semi, root))

            if self.sparse:
                if root in self._classes:
                    roots.append([root])
                    basses.append([(root + bass) % 12])
                else:
                    roots.append([fill])
                    basses.append([fill])
            else:
                if root in self._classes:
                    roots.extend(self.encoder.transform([[root]]))
                    basses.extend(self.encoder.transform([[(root + bass) % 12]]))
                else:
                    roots.extend(self.encoder.transform([[]]))
                    basses.extend(self.encoder.transform([[]]))

        pitches = np.asarray(pitches, dtype=np.bool)
        roots = np.asarray(roots, dtype=dtype)
        basses = np.asarray(basses, dtype=dtype)

        target_pitch = self.encode_intervals(duration, intervals, pitches)

        target_root = self.encode_intervals(duration, intervals, roots,
                                            multi=False,
                                            dtype=dtype,
                                            fill=fill)
        target_bass = self.encode_intervals(duration, intervals, basses,
                                            multi=False,
                                            dtype=dtype,
                                            fill=fill)

        if not self.sparse:
            target_root = _pad_nochord(target_root)
            target_bass = _pad_nochord(target_bass)

        return {'pitch': target_pitch,
                'root': target_root,
                'bass': target_bass}