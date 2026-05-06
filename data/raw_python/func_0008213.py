def critical(self, msg, indent=0, **kwargs):
        """invoke ``self.logger.critical``"""
        return self.logger.critical(self._indent(msg, indent), **kwargs)