def verifymessage(self, address, signature, message):
        """
        Verifies that a message has been signed by an address.

        Args:
          address (str): address claiming to have signed the message
          signature (str): ECDSA signature
          message (str): plaintext message which was signed

        Returns:
          bool: True if the address signed the message, False otherwise

        """
        verified = self.rpc.call("verifymessage", address, signature, message)
        self.logger.debug("Signature verified: %s" % str(verified))
        return verified