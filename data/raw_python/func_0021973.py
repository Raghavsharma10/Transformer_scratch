def safe_log_info(self, *info: str):
        """Log info failing silently on error"""
        self.__do_safe(lambda: self.logger.info(*info))