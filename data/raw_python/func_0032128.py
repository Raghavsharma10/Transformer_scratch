def maybeUpdate(self):
        """
        Check this cache entry and update it if any filesystem information has
        changed.
        """
        if self.wasModified():
            self.lastModified = self.filePath.getmtime()
            self.fileContents = self.filePath.getContent()
            self.hashValue = hashlib.sha1(self.fileContents).hexdigest()