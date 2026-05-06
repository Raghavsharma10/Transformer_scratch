def run_suite(self, suite, **kwargs):
        """This is the version from Django 1.7."""
        return self.test_runner(
            verbosity=self.verbosity,
            failfast=self.failfast,
            no_colour=self.no_colour,
        ).run(suite)