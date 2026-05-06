def start_virtual_display(self, width=1440, height=900,
                              colordepth=24, **kwargs):
        """Starts virtual display which will be
         destroyed after test execution will be end

        *Arguments:*
        - width: a width to be set in pixels
        - height: a height to be set in pixels
        - color_depth: a color depth to be used
        - kwargs: extra parameters

        *Example:*

        | Start Virtual Display |
        | Start Virtual Display | 1920 | 1080 |
        | Start Virtual Display | ${1920} | ${1080} | ${16} |
        """
        if self._display is None:
            logger.info("Using virtual display: '{0}x{1}x{2}'".format(
                        width, height, colordepth))

            self._display = Xvfb(int(width), int(height),
                                 int(colordepth), **kwargs)
            self._display.start()
            atexit.register(self._display.stop)