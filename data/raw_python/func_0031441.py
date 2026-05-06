def ORFs(self, openORFs=False):
        """
        Find all ORFs in our sequence.

        @param openORFs: If C{True} allow ORFs that do not have a start codon
            and/or do not have a stop codon.
        @return: A generator that yields AAReadORF instances that correspond
            to the ORFs found in the AA sequence.
        """

        # Return open ORFs to the left and right and closed ORFs within the
        # sequence.
        if openORFs:
            ORFStart = 0
            inOpenORF = True  # open on the left
            inORF = False

            for index, residue in enumerate(self.sequence):
                if residue == '*':
                    if inOpenORF:
                        if index:
                            yield AAReadORF(self, ORFStart, index, True, False)
                        inOpenORF = False
                    elif inORF:
                        if ORFStart != index:
                            yield AAReadORF(self, ORFStart, index,
                                            False, False)
                        inORF = False
                elif residue == 'M':
                    if not inOpenORF and not inORF:
                        ORFStart = index + 1
                        inORF = True

            # End of sequence. Yield the final ORF, open to the right, if
            # there is one and it has non-zero length.
            length = len(self.sequence)
            if inOpenORF and length > 0:
                yield AAReadORF(self, ORFStart, length, True, True)
            elif inORF and ORFStart < length:
                yield AAReadORF(self, ORFStart, length, False, True)

        # Return only closed ORFs.
        else:
            inORF = False

            for index, residue in enumerate(self.sequence):
                if residue == 'M':
                    if not inORF:
                        inORF = True
                        ORFStart = index + 1
                elif residue == '*':
                    if inORF:
                        if not ORFStart == index:
                            yield AAReadORF(self, ORFStart,
                                            index, False, False)
                        inORF = False