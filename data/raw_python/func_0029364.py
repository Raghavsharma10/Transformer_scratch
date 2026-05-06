def _process(self, scratch, filename, **kwargs):
        """Internal hook that marks reachable scripts before calling analyze.

        Returns data exactly as returned by the analyze method.

        """
        self.tag_reachable_scripts(scratch)
        return self.analyze(scratch, filename=filename, **kwargs)