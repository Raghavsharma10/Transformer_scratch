def error(self, message, rofi_args=None, **kwargs):
        """Show an error window.

        This method blocks until the user presses a key.

        Fullscreen mode is not supported for error windows, and if specified
        will be ignored.

        Parameters
        ----------
        message: string
            Error message to show.

        """
        rofi_args = rofi_args or []
        # Generate arguments list.
        args = ['rofi', '-e', message]
        args.extend(self._common_args(allow_fullscreen=False, **kwargs))
        args.extend(rofi_args)

        # Close any existing window and show the error.
        self._run_blocking(args)