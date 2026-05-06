def _show_coords_command(self, infile, outfile):
        '''Construct show-coords command'''
        command = 'show-coords -dTlro'

        if not self.coords_header:
            command += ' -H'

        return command + ' ' + infile + ' > ' + outfile