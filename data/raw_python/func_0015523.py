def nsname(self, uri: Union[str, URIRef]) -> str:
        """
        Return the 'ns:name' format of URI

        :param uri: URI to transform
        :return: nsname format of URI or straight URI if no mapping
        """
        uri = str(uri)
        nsuri = ""
        prefix = None
        for pfx, ns in self:
            nss = str(ns)
            if uri.startswith(nss) and len(nss) > len(nsuri):
                nsuri = nss
                prefix = pfx
        return (prefix.lower() + ':' + uri[len(nsuri):]) if prefix is not None else uri