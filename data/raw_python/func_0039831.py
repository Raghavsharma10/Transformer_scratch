def resolving(self, ref):
        """
        Context manager which resolves a JSON ``ref`` and enters the
        resolution scope of this ref.

        :argument str ref: reference to resolve

        """
        # print u"resol_scope: %s, ref: %s" % (self.resolution_scope, ref)
        full_uri = urljoin(self.resolution_scope, ref)
        uri, fragment = urldefrag(full_uri)
        if not uri:
            uri = self.base_uri

        if uri in self.store:
            document = self.store[uri]
        else:
            if (uri.startswith(u"file") or uri.startswith(u"File")):
                try:
                    document = self.resolve_local(full_uri, self.resolution_scope, ref)
                except Exception as exc:
                    raise RefResolutionError(exc)
            else:

                try:
                    document = self.resolve_remote(uri)
                except Exception as exc:
                    raise RefResolutionError(exc)

        old_base_uri, self.base_uri = self.base_uri, uri
        try:
            with self.in_scope(uri):
                yield self.resolve_fragment(document, fragment)
        finally:
            self.base_uri = old_base_uri