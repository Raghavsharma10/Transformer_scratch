def _setup_source_and_destination(self):
        """use the base class to setup the source and destinations but add to
        that setup the instantiation of the "new_crash_source" """
        super(FetchTransformSaveWithSeparateNewCrashSourceApp, self) \
            ._setup_source_and_destination()
        if self.config.new_crash_source.new_crash_source_class:
            self.new_crash_source = \
                self.config.new_crash_source.new_crash_source_class(
                    self.config.new_crash_source,
                    name=self.app_instance_name,
                    quit_check_callback=self.quit_check
                )
        else:
            # the configuration failed to provide a "new_crash_source", fall
            # back to tying the "new_crash_source" to the "source".
            self.new_crash_source = self.source