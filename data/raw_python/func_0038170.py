def signmessage(self, address, message):
        """Sign a message with the private key of an address.

        Cryptographically signs a message using ECDSA.  Since this requires
        an address's private key, the wallet must be unlocked first.

        Args:
          address (str): address used to sign the message
          message (str): plaintext message to which apply the signature

        Returns:
          str: ECDSA signature over the message

        """
        signature = self.rpc.call("signmessage", address, message)
        self.logger.debug("Signature: %s" % signature)
        return signature