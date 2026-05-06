def sign(self, message):
        """
        Generates a signature for the supplied message using NTLM2 Session Security
        Note: [MS-NLMP] Section 3.4.4
        The message signature for NTLM with extended session security is a 16-byte value that contains the following
        components, as described by the NTLMSSP_MESSAGE_SIGNATURE structure:
         - A 4-byte version-number value that is set to 1
         - The first eight bytes of the message's HMAC_MD5
         - The 4-byte sequence number (SeqNum)
        :param message: The message to be signed
        :return: The signature for supplied message
        """
        hmac_context = hmac.new(self.outgoing_signing_key)
        hmac_context.update(struct.pack('<i', self.outgoing_sequence) + message)

        # If a key exchange key is negotiated the first 8 bytes of the HMAC MD5 are encrypted with RC4
        if self.key_exchange:
            checksum = self.outgoing_seal.update(hmac_context.digest()[:8])
        else:
            checksum = hmac_context.digest()[:8]

        mac = _Ntlm2MessageSignature()
        mac['checksum'] = struct.unpack('<q', checksum)[0]
        mac['sequence'] = self.outgoing_sequence
        #logger.debug("Signing Sequence Number: %s", str(self.outgoing_sequence))

        # Increment the sequence number after signing each message
        self.outgoing_sequence += 1
        return str(mac)