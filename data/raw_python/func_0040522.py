def lint(self, targets):
        """Run linters in parallel and sort all results.

        Args:
            targets (list): List of files and folders to lint.
        """
        LinterRunner.targets = targets
        linters = self._config.get_linter_classes()
        with Pool() as pool:
            out_err_none = pool.map(LinterRunner.run, linters)
        out_err = [item for item in out_err_none if item is not None]
        stdout, stderr = zip(*out_err)
        return sorted(chain.from_iterable(stdout)), chain.from_iterable(stderr)