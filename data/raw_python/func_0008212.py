def error(self, msg, indent=0, **kwargs):
        """invoke ``self.logger.error``"""
        return self.logger.error(self._indent(msg, indent), **kwargs)