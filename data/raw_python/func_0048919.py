def info(self, message, *args, **kwargs):
        """alias to message at information level"""
        self.log("info", message, *args, **kwargs)