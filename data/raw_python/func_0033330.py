def _get_result_paths(self, data):
        """ Set the result paths
        """

        result = {}

        # OTU map (mandatory output)
        result['OtuMap'] = ResultPath(Path=self.Parameters['-O'].Value,
                                      IsWritten=True)

        # SumaClust will not produce any output file if the
        # input file was empty, so we create an empty
        # output file
        if not isfile(result['OtuMap'].Path):
            otumap_f = open(result['OtuMap'].Path, 'w')
            otumap_f.close()

        return result