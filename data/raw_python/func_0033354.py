def _get_result_paths(self, data):
        """ Set the result paths """

        result = {}

        result['Output'] = ResultPath(
            Path=self.Parameters['--output'].Value,
            IsWritten=self.Parameters['--output'].isOn())

        result['ClusterFile'] = ResultPath(
            Path=self.Parameters['--uc'].Value,
            IsWritten=self.Parameters['--uc'].isOn())

        result['PairwiseAlignments'] = ResultPath(
            Path=self.Parameters['--fastapairs'].Value,
            IsWritten=self.Parameters['--fastapairs'].isOn())

        return result