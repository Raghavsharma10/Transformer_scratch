def signed_raw(self) -> str:
        """
        If keys are None, returns the raw + current signatures
        If keys are present, returns the raw signed by these keys
        :return:
        """
        raw = self.raw()
        signed = "\n".join(self.signatures)
        signed_raw = raw + signed + "\n"
        return signed_raw