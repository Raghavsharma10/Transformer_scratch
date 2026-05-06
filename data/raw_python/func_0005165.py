def count(self, flag_message, padding=None, force=False):
        """ Log Level: :attr:COUNT

            @flag_message: time-like #float
            @padding: #str 'top', 'bottom' or 'all', adds a new line to the
                specified area with :func:padd
            @force: #bool whether or not to force the message to log in spite
                of the assigned log level

            ..
                from vital.debug import Logg
                logg = Logg(loglevel="v")

                logg("Total apps").count(3)
                # Total apps (3)

                logg().count([0, 1, 2, 3])
                # (4)
            ..
        """
        if self.should_log(self.COUNT) or force:
            flag_message = flag_message \
                if isinstance(flag_message, (int, float)) else \
                str(len(flag_message))
            self._print_message(
                flag_message=flag_message, padding=padding, reverse=True,
                color=colors.timing_color)