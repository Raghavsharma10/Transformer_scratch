def _delta_filter_command(self, infile, outfile):
        '''Construct delta-filter command'''
        command = 'delta-filter'

        if self.min_id is not None:
            command += ' -i ' + str(self.min_id)

        if self.min_length is not None:
            command += ' -l ' + str(self.min_length)

        return command + ' ' + infile + ' > ' + outfile