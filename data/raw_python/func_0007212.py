def set_log_level(self, level=logging.INFO):
        """Convenciene method to set the log level of the module given in the
        initializer of this class.

        :param level: and instance of ``logging.<level>``
        """
        mod, cls = self.parse_module_class()
        logging.getLogger(mod).setLevel(level)