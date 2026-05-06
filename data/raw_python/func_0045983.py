def getMessage(self):
        """Returns a colorized log message based on the log level.

        If the platform is windows the original message will be returned
        without colorization windows escape codes are crazy.

        :returns: ``str``
        """
        msg = str(self.msg)
        if self.args:
            msg = msg % self.args

        if platform.system().lower() == 'windows' or self.levelno < 10:
            return msg
        elif self.levelno >= 50:
            return utils.return_colorized(msg, 'critical')
        elif self.levelno >= 40:
            return utils.return_colorized(msg, 'error')
        elif self.levelno >= 30:
            return utils.return_colorized(msg, 'warn')
        elif self.levelno >= 20:
            return utils.return_colorized(msg, 'info')
        else:
            return utils.return_colorized(msg, 'debug')