def update_readme(self, content):
        """Update the readme descriptive metadata."""
        logger.debug("Updating readme")
        key = self.get_readme_key()

        # Back up old README content.
        backup_content = self.get_readme_content()
        backup_key = key + "-{}".format(
            timestamp(datetime.datetime.now())
        )
        logger.debug("README.yml backup key: {}".format(backup_key))
        self.put_text(backup_key, backup_content)

        self.put_text(key, content)