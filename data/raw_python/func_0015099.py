def address_verify(self, email, street, zip):
        """Shortcut for the AddressVerify method.

        ``email``::
            Email address of a PayPal member to verify.
            Maximum string length: 255 single-byte characters
            Input mask: ?@?.??
        ``street``::
            First line of the billing or shipping postal address to verify.

            To pass verification, the value of Street must match the first three
            single-byte characters of a postal address on file for the PayPal member.

            Maximum string length: 35 single-byte characters.
            Alphanumeric plus - , . ‘ # \
            Whitespace and case of input value are ignored.
        ``zip``::
            Postal code to verify.

            To pass verification, the value of Zip mustmatch the first five
            single-byte characters of the postal code of the verified postal
            address for the verified PayPal member.

            Maximumstring length: 16 single-byte characters.
            Whitespace and case of input value are ignored.
        """
        args = self._sanitize_locals(locals())
        return self._call('AddressVerify', **args)