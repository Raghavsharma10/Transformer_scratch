def save(self, content):
        """
        Save any given content to the instance file.
        :param content: (str or bytes)
        :return: (None)
        """
        # backup existing file if needed
        if os.path.exists(self.file_path) and not self.assume_yes:
            message = "Overwrite existing {}? (y/n) "
            if not confirm(message.format(self.filename)):
                self.backup()

        # write file
        self.output("Saving " + self.filename)
        with open(self.file_path, "wb") as handler:
            if not isinstance(content, bytes):
                content = bytes(content, "utf-8")
            handler.write(content)
        self.yeah("Done!")