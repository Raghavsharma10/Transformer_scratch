def validate_signature(self, signature, data, encoding='utf8'):
        """Validate the signature for the provided data.

        Args:
            signature (str or bytes or bytearray): Signature that was provided
                for the request.
            data (str or bytes or bytearray): Data string to validate against
                the signature.
            encoding (str, optional): If a string was provided for ``data`` or
                ``signature``, this is the character encoding.

        Returns:
            bool: Whether the signature is valid for the provided data.
        """

        if isinstance(data, string_types):
            data = bytearray(data, encoding)
        if isinstance(signature, string_types):
            signature = bytearray(signature, encoding)

        secret_key = bytearray(self.secret_key, 'utf8')
        hashed = hmac.new(secret_key, data, sha1)
        encoded = b64encode(hashed.digest())

        return encoded.strip() == signature.strip()