def simplify(self, chord):
        '''Simplify a chord string down to the vocabulary space'''
        # Drop inversions
        chord = re.sub(r'/.*$', r'', chord)
        # Drop any additional or suppressed tones
        chord = re.sub(r'\(.*?\)', r'', chord)
        # Drop dangling : indicators
        chord = re.sub(r':$', r'', chord)

        # Encode the chord
        root, pitches, _ = mir_eval.chord.encode(chord)

        # Build the query
        # To map the binary vector pitches down to bit masked integer,
        # we just dot against powers of 2
        P = 2**np.arange(12, dtype=int)
        query = self.mask_ & pitches[::-1].dot(P)

        if root < 0 and chord[0].upper() == 'N':
            return 'N'
        if query not in QUALITIES:
            return 'X'

        return '{}:{}'.format(PITCHES[root], QUALITIES[query])