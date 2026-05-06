def safe_log_error(self, error: Exception, *info: str):
        """Log error failing silently on error"""
        self.__do_safe(lambda: self.logger.error(error, *info))