def run(self):
        """Main run loop for all components.

        Performs initial handshake with Storm and reads Tuples handing them off
        to subclasses.  Any exceptions are caught and logged back to Storm
        prior to the Python process exiting.

        .. warning::

            Subclasses should **not** override this method.
        """
        storm_conf, context = self.read_handshake()
        self._setup_component(storm_conf, context)
        self.initialize(storm_conf, context)
        while True:
            try:
                self._run()
            except StormWentAwayError:
                log.info("Exiting because parent Storm process went away.")
                self._exit(2)
            except Exception as e:
                log_msg = "Exception in {}.run()".format(self.__class__.__name__)
                exc_info = sys.exc_info()
                try:
                    self.logger.error(log_msg, exc_info=True)
                    self._handle_run_exception(e)
                except StormWentAwayError:
                    log.error(log_msg, exc_info=exc_info)
                    log.info("Exiting because parent Storm process went away.")
                    self._exit(2)
                except:
                    log.error(log_msg, exc_info=exc_info)
                    log.error(
                        "While trying to handle previous exception...",
                        exc_info=sys.exc_info(),
                    )

                if self.exit_on_exception:
                    self._exit(1)