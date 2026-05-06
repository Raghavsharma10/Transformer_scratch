def on_modified(self, event):
        """
        Modified config file handler.

        If a config file is modified, the yaml contents are parsed and the
        new results are validated by the target class.  Once validated, the
        new config is passed to the on_update callback.
        """
        if os.path.isdir(event.src_path):
            return

        logger.debug("file modified: %s", event.src_path)

        name = self.file_name(event)

        try:
            config = yaml.load(open(event.src_path))
            self.target_class.from_config(name, config)
        except Exception:
            logger.exception(
                "Error when loading updated config file %s", event.src_path,
            )
            return

        self.on_update(self.target_class, name, config)