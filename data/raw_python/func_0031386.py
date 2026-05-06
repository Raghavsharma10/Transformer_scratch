def _cigarString(self, output):
        """
        Return a cigar string of aligned sequences.

        @param output: a C{tup} of strings (align1, align, align2)
        @return: a C{str} containing the cigar string. Eg with input:
            'GGCCCGCA' and 'GG-CTGCA', return 2=1D1=1X3=
        """
        cigar = []
        count = 0
        align1 = output[0]
        align2 = output[2]
        for nt1, nt2 in zip(align1, align2):
            if nt1 == nt2:
                cigar.append('=')
            elif nt1 == '-':
                cigar.append('I')
            elif nt2 == '-':
                cigar.append('D')
            else:
                cigar.append('X')
        # Initially create a list of characters,
        # eg ['=', '=', 'D', '=', 'X', '=', '=', '=']
        cigar.append('*')
        # Append an arbitrary character to ensure parsing below functions
        cigarString = ''
        previousCharacter = ''
        count = 0
        first = True
        for character in cigar:
            if first:
                previousCharacter = character
                count += 1
                first = False
            else:
                if character == previousCharacter:
                    count += 1
                else:
                    cigarString += (str(count) + str(previousCharacter))
                    count = 1
                previousCharacter = character
        return cigarString