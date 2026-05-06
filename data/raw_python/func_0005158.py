def log(self, flag_message=None, padding=None, color=None, force=False):
        """ Log Level: :attr:LOG

            @flag_message: #str flags the message with the given text
                using :func:flag
            @padding: #str 'top', 'bottom' or 'all', adds a new line to the
                specified area with :func:padd
            @color: #str colorizes @flag_message using :func:colorize
            @force: #bool whether or not to force the message to log in spite
                of the assigned log level

            ..
                from vital.debug import Logg
                logg = Logg(loglevel="v")

                logg("World").log("Hello")
                # (Hello) World

                logg("Hello world").log()
                # Hello world
            ..
        """
        if self.should_log(self.LOG) or force:
            self._print_message(
                flag_message=flag_message, color=color or colors.bold,
                padding=padding)