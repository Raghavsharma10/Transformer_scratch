def _get_result_paths(self, data):
        """ Set the result paths """

        result = {}

        result['Output'] = ResultPath(
            Path=self.Parameters['--output'].Value,
            IsWritten=self.Parameters['--output'].isOn())

        result['ClusterFile'] = ResultPath(
            Path=self.Parameters['--uc'].Value,
            IsWritten=self.Parameters['--uc'].isOn())

        # uchime 3-way global alignments
        result['Output_aln'] = ResultPath(
            Path=self.Parameters['--uchimealns'].Value,
            IsWritten=self.Parameters['--uchimealns'].isOn())

        # uchime tab-separated format
        result['Output_tabular'] = ResultPath(
            Path=self.Parameters['--uchimeout'].Value,
            IsWritten=self.Parameters['--uchimeout'].isOn())

        # chimeras fasta file output
        result['Output_chimeras'] = ResultPath(
            Path=self.Parameters['--chimeras'].Value,
            IsWritten=self.Parameters['--chimeras'].isOn())

        # nonchimeras fasta file output
        result['Output_nonchimeras'] = ResultPath(
            Path=self.Parameters['--nonchimeras'].Value,
            IsWritten=self.Parameters['--nonchimeras'].isOn())

        # log file
        result['LogFile'] = ResultPath(
            Path=self.Parameters['--log'].Value,
            IsWritten=self.Parameters['--log'].isOn())

        return result