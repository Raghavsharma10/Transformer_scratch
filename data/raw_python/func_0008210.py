def info(self, msg, indent=0, **kwargs):
        """invoke ``self.info.debug``"""
        return self.logger.info(self._indent(msg, indent), **kwargs)