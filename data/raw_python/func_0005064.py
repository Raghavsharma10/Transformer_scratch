def run(self):
        # type: () -> bool
        """ Run all linters and report results.

        Returns:
            bool: **True** if all checks were successful, **False** otherwise.
        """
        with util.timed_block() as t:
            files = self._collect_files()

        log.info("Collected <33>{} <32>files in <33>{}s".format(
            len(files), t.elapsed_s
        ))
        if self.verbose:
            for p in files:
                log.info("  <0>{}", p)

        # No files to lint - return success if empty runs are allowed.
        if not files:
            return self.allow_empty

        with util.timed_block() as t:
            results = self._run_checks(files)

        log.info("Code checked in <33>{}s", t.elapsed_s)

        success = True
        for name, retcodes in results.items():
            if any(x != 0 for x in retcodes):
                success = False
                log.err("<35>{} <31>failed with: <33>{}".format(
                    name, retcodes
                ))

        return success