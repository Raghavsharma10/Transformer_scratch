def _get_result_paths(self, data):
        """Captures SeqPrep output.

        """
        result = {}

        # Always output:
        result['UnassembledReads1'] = ResultPath(Path=
                                                 self._unassembled_reads1_out_file_name(
                                                 ),
                                                 IsWritten=True)
        result['UnassembledReads2'] = ResultPath(Path=
                                                 self._unassembled_reads2_out_file_name(
                                                 ),
                                                 IsWritten=True)

        # optional output, so we check for each
        # check for assembled reads file
        if self.Parameters['-s'].isOn():
            result['Assembled'] = ResultPath(Path=
                                             self._assembled_out_file_name(),
                                             IsWritten=True)

        # check for discarded (unassembled) reads1 file
        if self.Parameters['-3'].isOn():
            result['Reads1Discarded'] = ResultPath(Path=
                                                   self._discarded_reads1_out_file_name(
                                                   ),
                                                   IsWritten=True)

        # check for discarded (unassembled) reads2 file
        if self.Parameters['-4'].isOn():
            result['Reads2Discarded'] = ResultPath(Path=
                                                   self._discarded_reads2_out_file_name(
                                                   ),
                                                   IsWritten=True)

        # check for pretty-alignment file
        if self.Parameters['-E'].isOn():
            result['PrettyAlignments'] = ResultPath(Path=
                                                    self._pretty_alignment_out_file_name(
                                                    ),
                                                    IsWritten=True)

        return result