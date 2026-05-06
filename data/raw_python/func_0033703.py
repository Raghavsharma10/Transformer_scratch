def _compile_mothur_script(self):
        """Returns a Mothur batch script as a string"""
        fasta = self._input_filename

        required_params = ["reference", "taxonomy"]
        for p in required_params:
            if self.Parameters[p].Value is None:
                raise ValueError("Must provide value for parameter %s" % p)
        optional_params = ["ksize", "cutoff", "iters"]
        args = self._format_function_arguments(
            required_params + optional_params)
        script = '#classify.seqs(fasta=%s, %s)' % (fasta, args)
        return script