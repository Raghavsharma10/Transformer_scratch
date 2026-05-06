def info(self, verbose=None):
        """ Prints and formats the results of the timing
            @_print: #bool whether or not to print out to terminal
            @verbose: #bool True if you'd like to print the individual timing
                results in additions to the comparison results
        """
        if self.name:
            flag(bold(self.name))
        flag("Results after {} intervals".format(
            bold(self.num_intervals, close=False)),
            colors.notice_color, padding="top")
        line("‒")
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            for result in self._callable_results:
                result.info()
                line()
        diffs = [
            (i, result.mean)
            for i, result in enumerate(self._callable_results)
            if result.mean]
        ranking = [
            (i, self._callable_results[i].format_time(r))
            for i, r in sorted(diffs, key=lambda x: x[1])]
        max_rlen = len(str(len(ranking)))+2
        max_rlen2 = max(len(r) for i, r in ranking)+1
        best = self._callable_results[ranking[0][0]].mean
        for idx, (i, rank) in enumerate(ranking, 1):
            _obj_name = Look(self._callables[i]).objname()
            pct = "".rjust(10) if idx == 1 else \
                self._pct_diff(best, self._callable_results[i].mean)
            print(
                ("#"+str(idx)+" ¦").rjust(max_rlen), rank.rjust(max_rlen2),
                pct, "{}".format(_obj_name))
        line("‒", padding="bottom")