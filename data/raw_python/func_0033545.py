def _get_result_paths(self, data):
        """Capture fastq-join output.

        Three output files are produced, in the form of
            outputjoin : assembled paired reads
            outputun1 : unassembled reads_1
            outputun2 : unassembled reads_2

        If a barcode / mate-pairs file is also provided then the following
        additional files are output:
            outputjoin2
            outputun3

        If a verbose stitch length report (-r) is chosen to be written by the
        user then use a user specified filename.
        """
        output_path = self._get_output_path()

        result = {}

        # always output:
        result['Assembled'] = ResultPath(Path=output_path + 'join',
                                         IsWritten=True)
        result['UnassembledReads1'] = ResultPath(Path=output_path + 'un1',
                                                 IsWritten=True)
        result['UnassembledReads2'] = ResultPath(Path=output_path + 'un2',
                                                 IsWritten=True)

        # check if stitch report is requested:
        stitch_path = self._get_stitch_report_path()
        if stitch_path:
            result['Report'] = ResultPath(Path=stitch_path,
                                          IsWritten=True)

        # Check if mate file / barcode file is present.
        # If not, return result
        # We need to check this way becuase there are no infile parameters.
        mate_path_string = output_path + 'join2'
        mate_unassembled_path_string = output_path + 'un3'
        if os.path.exists(mate_path_string) and \
                os.path.exists(mate_unassembled_path_string):
            result['Mate'] = ResultPath(Path=mate_path_string,
                                        IsWritten=True)
            result['MateUnassembled'] = ResultPath(Path=
                                                   mate_unassembled_path_string,
                                                   IsWritten=True)
        else:
            pass
        return result