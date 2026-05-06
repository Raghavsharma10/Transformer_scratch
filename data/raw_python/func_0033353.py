def _input_as_parameters(self, data):
        """ Set the input path (a fasta filepath)
        """
        # The list of values which can be passed on a per-run basis
        allowed_values = ['--input', '--uc', '--fastapairs',
                          '--uc2clstr', '--output', '--mergesort']

        unsupported_parameters = set(data.keys()) - set(allowed_values)
        if unsupported_parameters:
            raise ApplicationError(
                "Unsupported parameter(s) passed when calling uclust: %s" %
                ' '.join(unsupported_parameters))

        for v in allowed_values:
            # turn the parameter off so subsequent runs are not
            # affected by parameter settings from previous runs
            self.Parameters[v].off()
            if v in data:
                # turn the parameter on if specified by the user
                self.Parameters[v].on(data[v])

        return ''