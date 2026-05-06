def timing(self, flag_message, padding=None, force=False):
        """ Log Level: :attr:TIMING

            @flag_message: time-like #float
            @padding: #str 'top', 'bottom' or 'all', adds a new line to the
                specified area with :func:padd
            @force: #bool whether or not to force the message to log in spite
                of the assigned log level

            ..
                from vital.debug import Logg
                logg = Logg(loglevel="v")

                logg("Finished in").timing(0.908)
                # Finished in (908.0ms)

                logg().timing(0.908)
                # (908.0ms)
            ..
        """
        if self.should_log(self.TIMING) or force:
            self._print_message(
                flag_message=Timer.format_time(flag_message), padding=padding,
                reverse=True, color=colors.timing_color)