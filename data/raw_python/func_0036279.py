def on_deleted(self, event):
        """
        Deleted config file handler.

        Simply fires the on_delete callback with the name of the deleted item.
        """
        logger.debug("file removed: %s", event.src_path)
        name = self.file_name(event)

        self.on_delete(self.target_class, name)