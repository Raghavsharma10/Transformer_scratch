def mime(self):
        """
        Returns the finalised mime object, after
        applying the internal headers. Usually this
        is not to be overriden.
        """
        mime = self.mime_object()
        self.headers.prepare(mime)
        return mime