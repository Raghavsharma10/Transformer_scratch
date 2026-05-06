def _write_script(self, script_name, ref, qry, outfile):
        '''Write commands into a bash script'''
        f = pyfastaq.utils.open_file_write(script_name)
        print(self._nucmer_command(ref, qry, 'p'), file=f)
        print(self._delta_filter_command('p.delta', 'p.delta.filter'), file=f)
        print(self._show_coords_command('p.delta.filter', outfile), file=f)
        if self.show_snps:
            print(self._show_snps_command('p.delta.filter', outfile + '.snps'), file=f)
        pyfastaq.utils.close(f)