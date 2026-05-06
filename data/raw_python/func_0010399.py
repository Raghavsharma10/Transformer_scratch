def filename_unsafe(self):
        """The filename from the Content-Disposition header.

        If a location was passed at instanciation, the basename
        from that may be used as a fallback. Otherwise, this may
        be the None value.

        On safety:
            This property records the intent of the sender.

            You shouldn't use this sender-controlled value as a filesystem
        path, it can be insecure. Serving files with this filename can be
        dangerous as well, due to a certain browser using the part after the
        dot for mime-sniffing.
        Saving it to a database is fine by itself though.
        """

        if 'filename*' in self.assocs:
            return self.assocs['filename*'].string
        elif 'filename' in self.assocs:
            # XXX Reject non-ascii (parsed via qdtext) here?
            return self.assocs['filename']
        elif self.location is not None:
            return posixpath.basename(self.location_path.rstrip('/'))