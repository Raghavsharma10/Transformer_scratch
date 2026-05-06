def verify(self, message, signature):
        """
        Verified the signature attached to the supplied message using NTLM2 Session Security
        :param message: The message whose signature will verified
        :return: True if the signature is valid, otherwise False
        """
        # Parse the signature header
        mac = _Ntlm2MessageSignature()
        mac.from_string(signature)

        # validate the sequence
        if mac['sequence'] != self.incoming_sequence:
            raise Exception("The message was not received in the correct sequence.")

        # extract the supplied checksum
        checksum = struct.pack('<q', mac['checksum'])
        if self.key_exchange:
            checksum = self.incoming_seal.update(checksum)

        # calculate the expected checksum for the message
        hmac_context = hmac.new(self.incoming_signing_key)
        hmac_context.update(struct.pack('<i', self.incoming_sequence) + message)
        expected_checksum = hmac_context.digest()[:8]

        # validate the supplied checksum is correct
        if checksum != expected_checksum:
            raise Exception("The message has been altered")

        #logger.debug("Verify Sequence Number: %s", AsHex(self.outgoing_sequence))
        self.incoming_sequence += 1