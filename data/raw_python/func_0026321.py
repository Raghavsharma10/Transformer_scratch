def entries(self):
        """A list of :class:`PasswordEntry` objects."""
        timer = Timer()
        passwords = []
        logger.info("Scanning %s ..", format_path(self.directory))
        listing = self.context.capture("find", "-type", "f", "-name", "*.gpg", "-print0")
        for filename in split(listing, "\0"):
            basename, extension = os.path.splitext(filename)
            if extension == ".gpg":
                # We use os.path.normpath() to remove the leading `./' prefixes
                # that `find' adds because it searches the working directory.
                passwords.append(PasswordEntry(name=os.path.normpath(basename), store=self))
        logger.verbose("Found %s in %s.", pluralize(len(passwords), "password"), timer)
        return natsort(passwords, key=lambda e: e.name)