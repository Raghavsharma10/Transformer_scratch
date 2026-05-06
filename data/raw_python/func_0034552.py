def _weaken_key(flags, key):
        """
        NOTE: Key weakening in NTLM2 (Extended Session Security) is performed simply by truncating the master key (or
        secondary master key, if key exchange is performed) to the appropriate length. 128-bit keys are supported under
        NTLM2. In this case, the master key is used directly in the generation of subkeys (with no weakening performed).
        :param flags: The negotiated NTLM flags
        :return: The 16-byte key to be used to sign messages
        """
        if flags & NegotiateFlag.NTLMSSP_KEY_128:
            return key
        if flags & NegotiateFlag.NTLMSSP_NEGOTIATE_56:
            return key[:7]
        else:
            return key[:5]