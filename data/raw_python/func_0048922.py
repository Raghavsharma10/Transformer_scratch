def warning(self, message, *args, **kwargs):
        """alias to message at warning level"""
        self.log("warn", message, *args, **kwargs)