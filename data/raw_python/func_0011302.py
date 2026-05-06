def exit_with_error(self, error, **kwargs):
        """Report an error and exit.

        This raises a SystemExit exception to ask the interpreter to quit.

        Parameters
        ----------
        error: string
            The error to report before quitting.

        """
        self.error(error, **kwargs)
        raise SystemExit(error)