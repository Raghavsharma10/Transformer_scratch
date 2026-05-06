def sign(self, keys: list) -> None:
        """
        Sign the current document.
        Warning : current signatures will be replaced with the new ones.

        :param keys: List of libnacl key instances
        :return:
        """
        if not isinstance(self.identity, Identity):
            raise MalformedDocumentError("Can not return full revocation document created from inline")

        self.signatures = []
        for key in keys:
            signing = base64.b64encode(key.signature(bytes(self.raw(), 'ascii')))
            self.signatures.append(signing.decode("ascii"))