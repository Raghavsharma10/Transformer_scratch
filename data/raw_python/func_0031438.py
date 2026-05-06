def translations(self):
        """
        Yield all six translations of a nucleotide sequence.

        @return: A generator that produces six L{TranslatedRead} instances.
        """
        rc = self.reverseComplement().sequence
        for reverseComplemented in False, True:
            for frame in 0, 1, 2:
                seq = rc if reverseComplemented else self.sequence
                # Get the suffix of the sequence for translation. I.e.,
                # skip 0, 1, or 2 initial bases, depending on the frame.
                # Note that this makes a copy of the sequence, which we can
                # then safely append 'N' bases to to adjust its length to
                # be zero mod 3.
                suffix = seq[frame:]
                lengthMod3 = len(suffix) % 3
                if lengthMod3:
                    suffix += ('NN' if lengthMod3 == 1 else 'N')
                yield TranslatedRead(self, translate(suffix), frame,
                                     reverseComplemented)