def warning(self, msg, indent=0, **kwargs):
        """invoke ``self.logger.warning``"""
        return self.logger.warning(self._indent(msg, indent), **kwargs)