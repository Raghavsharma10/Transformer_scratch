def signed_raw(self) -> str:
        """
        Return Revocation signed raw document string

        :return:
        """
        if not isinstance(self.identity, Identity):
            raise MalformedDocumentError("Can not return full revocation document created from inline")

        raw = self.raw()
        signed = "\n".join(self.signatures)
        signed_raw = raw + signed + "\n"
        return signed_raw