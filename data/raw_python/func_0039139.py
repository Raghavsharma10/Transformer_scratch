def scan(self, filetypes=None):
        """Iterates over the content of the disk and queries VirusTotal
        to determine whether it's malicious or not.

        filetypes is a list containing regular expression patterns.
        If given, only the files which type will match with one or more of
        the given patterns will be queried against VirusTotal.

        For each file which is unknown by VT or positive to any of its engines,
        the method yields a namedtuple:

        VTReport(path        -> C:\\Windows\\System32\\infected.dll
                 hash        -> ab231...
                 detections) -> dictionary engine -> detection

        Files unknown by VirusTotal will contain the string 'unknown'
        in the detection field.

        """
        self.logger.debug("Scanning FS content.")
        checksums = self.filetype_filter(self._filesystem.checksums('/'),
                                         filetypes=filetypes)

        self.logger.debug("Querying %d objects to VTotal.", len(checksums))

        for files in chunks(checksums, size=self.batchsize):
            files = dict((reversed(e) for e in files))
            response = vtquery(self._apikey, files.keys())

            yield from self.parse_response(files, response)