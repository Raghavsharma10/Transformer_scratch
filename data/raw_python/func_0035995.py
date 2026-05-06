def _run(self):
        """Run the worker function with some custom exception handling."""
        try:
            # Run the worker
            self.worker()
        except SystemExit as ex:
            # sys.exit() was called
            if isinstance(ex.code, int):
                if ex.code is not None and ex.code != 0:
                    # A custom exit code was specified
                    self._shutdown(
                        'Exiting with non-zero exit code {exitcode}'.format(
                            exitcode=ex.code),
                        ex.code)
            else:
                # A message was passed to sys.exit()
                self._shutdown(
                    'Exiting with message: {msg}'.format(msg=ex.code), 1)
        except Exception as ex:
            if self.detach:
                self._shutdown('Dying due to unhandled {cls}: {msg}'.format(
                    cls=ex.__class__.__name__, msg=str(ex)), 127)
            else:
                # We're not detached so just raise the exception
                raise

        self._shutdown('Shutting down normally')