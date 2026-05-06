def getMd5Checksum(self):
        """
        Returns the MD5 checksum for this reference set. This checksum is
        calculated by making a list of `Reference.md5checksum` for all
        `Reference`s in this set. We then sort this list, and take the
        MD5 hash of all the strings concatenated together.
        """
        references = sorted(
            self.getReferences(),
            key=lambda ref: ref.getMd5Checksum())
        checksums = ''.join([ref.getMd5Checksum() for ref in references])
        md5checksum = hashlib.md5(checksums).hexdigest()
        return md5checksum