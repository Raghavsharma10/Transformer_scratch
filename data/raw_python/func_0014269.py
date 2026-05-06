def handle_exceptions(self, verbose=True):
        """
        Handle Ctrl+C and accidental exceptions and attempt to save
        the current state of the simulation
        """
        try:
            yield
        except (KeyboardInterrupt, Exception) as ex:
            if not self.attempt_rescue:
                raise ex
            if isinstance(ex, KeyboardInterrupt):
                reraise = False
                answer = timed_input('\n\nDo you want to save current state? (y/N): ')
                if answer and answer.lower() not in ('y', 'yes'):
                    if verbose:
                        sys.exit('Ok, bye!')
            else:
                reraise = True
                logger.error('\n\nAn error occurred: %s', ex)
            if verbose:
                logger.info('Saving state...')
            try:
                self.backup_simulation()
            except Exception:
                if verbose:
                    logger.error('FAILED :(')
            else:
                if verbose:
                    logger.info('SUCCESS!')
            finally:
                if reraise:
                    raise ex
                sys.exit()