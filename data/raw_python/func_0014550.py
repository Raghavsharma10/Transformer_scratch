def _pkcs1imify(self, data):
        """
        turn a 20-byte SHA1 hash into a blob of data as large as the key's N,
        using PKCS1's \"emsa-pkcs1-v1_5\" encoding.  totally bizarre.
        """
        SHA1_DIGESTINFO = '\x30\x21\x30\x09\x06\x05\x2b\x0e\x03\x02\x1a\x05\x00\x04\x14'
        size = len(util.deflate_long(self.n, 0))
        filler = '\xff' * (size - len(SHA1_DIGESTINFO) - len(data) - 3)
        return '\x00\x01' + filler + '\x00' + SHA1_DIGESTINFO + data