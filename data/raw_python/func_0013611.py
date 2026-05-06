def populateFromFile(self, dataUrl):
        """
        Populates the instance variables of this ReferencSet from the
        data URL.
        """
        self._dataUrl = dataUrl
        fastaFile = self.getFastaFile()
        for referenceName in fastaFile.references:
            reference = HtslibReference(self, referenceName)
            # TODO break this up into chunks and calculate the MD5
            # in bits (say, 64K chunks?)
            bases = fastaFile.fetch(referenceName)
            md5checksum = hashlib.md5(bases).hexdigest()
            reference.setMd5checksum(md5checksum)
            reference.setLength(len(bases))
            self.addReference(reference)