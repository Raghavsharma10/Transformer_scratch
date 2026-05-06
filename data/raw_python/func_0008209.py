def debug(self, msg, indent=0, **kwargs):
        """invoke ``self.logger.debug``"""
        return self.logger.debug(self._indent(msg, indent), **kwargs)