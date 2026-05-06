def create_logger(self):
        """Generates a logger instance from the singleton"""
        name = "bors"
        if hasattr(self, "name"):
            name = self.name

        self.log = logging.getLogger(name)
        try:
            lvl = self.conf.get_log_level()
        except AttributeError:
            lvl = self.context.get("log_level", None)

        self.log.setLevel(getattr(logging, lvl, logging.INFO))