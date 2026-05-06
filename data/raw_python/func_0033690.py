def _compile_mothur_script(self):
        """Returns a Mothur batch script as a string"""
        def format_opts(*opts):
            """Formats a series of options for a Mothur script"""
            return ', '.join(filter(None, map(str, opts)))
        vars = {
            'in': self._input_filename,
            'unique': self._derive_unique_path(),
            'dist': self._derive_dist_path(),
            'names': self._derive_names_path(),
            'cluster_opts': format_opts(
                self.Parameters['method'],
                self.Parameters['cutoff'],
                self.Parameters['precision'],
            ),
        }
        script = (
            '#'
            'unique.seqs(fasta=%(in)s); '
            'dist.seqs(fasta=%(unique)s); '
            'read.dist(column=%(dist)s, name=%(names)s); '
            'cluster(%(cluster_opts)s)' % vars
        )
        return script