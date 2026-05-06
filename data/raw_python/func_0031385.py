def _fillAndTraceback(self, table):
        """
        Perform Local Alignment according to Smith-Waterman Algorithm.
        Fills the table and then traces back from the highest score.
        NB left = deletion and up = insertion wrt seq1
        """
        # Fill
        max_score = 0
        max_row = 0
        max_col = 0

        for row in range(1, len(self.seq2Seq) + 1):
            for col in range(1, len(self.seq1Seq) + 1):
                # Calculate match score
                letter1 = self.seq1Seq[col - 1]
                letter2 = self.seq2Seq[row - 1]
                if letter1 == letter2:
                    diagonal_score = (table[row - 1][col - 1]['score'] +
                                      self.match)
                else:
                    diagonal_score = (table[row - 1][col - 1]['score'] +
                                      self.mismatch)

                ins_run = table[row - 1][col]['ins']
                del_run = table[row][col - 1]['del']

                # Calculate gap scores ensuring extension is not > 0
                if table[row - 1][col]['ins'] <= 0:
                    ins_score = table[row - 1][col]['score'] + self.gapOpen
                else:
                    if self.gapExtend + ins_run * self.gapExtendDecay <= 0.0:
                        ins_score = (table[row - 1][col]['score'] +
                                     self.gapExtend +
                                     ins_run * self.gapExtendDecay)
                    else:
                        ins_score = table[row - 1][col]['score']

                if table[row - 1][col]['del'] <= 0:
                    del_score = table[row][col - 1]['score'] + self.gapOpen
                else:
                    if self.gapExtend + del_run * self.gapExtendDecay <= 0.0:
                        del_score = (table[row][col - 1]['score'] +
                                     self.gapExtend +
                                     del_run * self.gapExtendDecay)
                    else:
                        del_score = table[row][col - 1]['score']

                # Choose best score
                if diagonal_score <= 0 and ins_score <= 0 and del_score <= 0:
                    table[row][col] = {'score': 0, 'pointer': None, 'ins': 0,
                                       'del': 0}
                else:
                    if diagonal_score >= ins_score:
                        if diagonal_score >= del_score:  # diag lef/up
                            diagonal = {'score': diagonal_score,
                                        'pointer': 'diagonal', 'ins': 0,
                                        'del': 0}
                            table[row][col] = diagonal
                        else:  # lef diag/up
                            deletion = {'score': del_score, 'pointer': 'del',
                                        'ins': 0, 'del': del_run + 1}
                            table[row][col] = deletion
                    else:  # up diag
                        if ins_score >= del_score:  # up diag/lef
                            insertion = {'score': ins_score, 'pointer': 'ins',
                                         'ins': ins_run + 1, 'del': 0}
                            table[row][col] = insertion
                        else:  # lef up diag
                            deletion = {'score': del_score, 'pointer': 'del',
                                        'ins': 0, 'del': del_run + 1}
                            table[row][col] = deletion

                # Set max score - is this the best way of getting max score
                # considering how the for loop iterates through the matrix?
                if table[row][col]['score'] >= max_score:
                    max_row = row
                    max_col = col
                    max_score = table[row][col]['score']

        # Traceback
        indexes = {'max_row': max_row, 'max_col': max_col}
        align1 = ''
        align2 = ''
        align = ''

        current_row = max_row
        current_col = max_col

        while True:
            arrow = table[current_row][current_col]['pointer']
            if arrow is None:
                min_row = current_row + 1
                min_col = current_col + 1
                break
            elif arrow == 'diagonal':
                align1 += self.seq1Seq[current_col - 1]
                align2 += self.seq2Seq[current_row - 1]
                if self.seq1Seq[current_col - 1] == self.seq2Seq[
                        current_row - 1]:
                    align += '|'
                else:
                    align += ' '
                current_row -= 1
                current_col -= 1
            elif arrow == 'del':
                align1 += self.seq1Seq[current_col - 1]
                align2 += '-'
                align += ' '
                current_col -= 1
            elif arrow == 'ins':
                align1 += '-'
                align2 += self.seq2Seq[current_row - 1]
                align += ' '
                current_row -= 1
            else:
                raise ValueError('Invalid pointer: %s' % arrow)

        indexes['min_row'] = min_row
        indexes['min_col'] = min_col
        align1 = align1[::-1]
        align2 = align2[::-1]
        align = align[::-1]

        if len(align1) != len(align2):
            raise ValueError(
                'Lengths of locally aligned sequences differ (%d != %d).' % (
                    len(align1), len(align2)))

        return ([align1, align, align2], indexes)