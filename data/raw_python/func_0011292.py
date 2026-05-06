def status(self, message, rofi_args=None, **kwargs):
        """Show a status message.

        This method is non-blocking, and intended to give a status update to
        the user while something is happening in the background.

        To close the window, either call the close() method or use any of the
        display methods to replace it with a different window.

        Fullscreen mode is not supported for status messages and if specified
        will be ignored.

        Parameters
        ----------
        message: string
            Progress message to show.

        """
        rofi_args = rofi_args or []
        # Generate arguments list.
        args = ['rofi', '-e', message]
        args.extend(self._common_args(allow_fullscreen=False, **kwargs))
        args.extend(rofi_args)

        # Update the status.
        self._run_nonblocking(args)