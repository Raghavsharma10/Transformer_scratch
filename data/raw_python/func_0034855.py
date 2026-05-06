def exclude_functions(self, *funcs):
        """
        Excludes the contributions from the following functions.
        """
        for f in funcs:
            f.exclude = True
        run_time_s = sum(0 if s.exclude else s.own_time_s for s in self.stats)
        cProfileFuncStat.run_time_s = run_time_s