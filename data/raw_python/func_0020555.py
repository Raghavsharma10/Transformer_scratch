def __flags(self):
        """
        Internal method. Turns arguments into flags.
        """
        flags = []
        if self._capture:
            flags.append("-capture")
        if self._spy:
            flags.append("-spy")
        if self._dbpath:
            flags += ["-db-path", self._dbpath]
            flags += ["-db", "boltdb"]
        else:
            flags += ["-db", "memory"]
        if self._synthesize:
            assert(self._middleware)
            flags += ["-synthesize"]
        if self._simulation:
            flags += ["-import", self._simulation]
        if self._proxyPort:
            flags += ["-pp", str(self._proxyPort)]
        if self._adminPort:
            flags += ["-ap", str(self._adminPort)]
        if self._modify:
            flags += ["-modify"]
        if self._verbose:
            flags += ["-v"]
        if self._dev:
            flags += ["-dev"]
        if self._metrics:
            flags += ["-metrics"]
        if self._auth:
            flags += ["-auth"]
        if self._middleware:
            flags += ["-middleware", self._middleware]
        if self._cert:
            flags += ["-cert", self._cert]
        if self._certName:
            flags += ["-cert-name", self._certName]
        if self._certOrg:
            flags += ["-cert-org", self._certOrg]
        if self._destination:
            flags += ["-destination", self._destination]
        if self._key:
            flags += ["-key", self._key]
        if self._dest:
            for i in range(len(self._dest)):
                flags += ["-dest", self._dest[i]]
        if self._generateCACert:
            flags += ["-generate-ca-cert"]
        if not self._tlsVerification:
            flags += ["-tls-verification", "false"]

        logging.debug("flags:" + str(flags))
        return flags