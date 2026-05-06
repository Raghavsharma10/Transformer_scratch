def decrypt(self, recipient_key):
        """
        recipient_key: UserLock with a private key part.
        returns: filename, decrypted file contents
        """
        if recipient_key.private_key is None:
            raise ValueError("Cannot decrypt with this key; no private key part found.")
        header = MiniLockHeader(self.header)
        # Create ephemeral public key for authenticated decryption of metadata.
        # TODO: Future-proof this by making it try to decrypt a b58 ephem ID if available?
        decryptInfo = header.decrypt(recipient_key)
        file_info = decryptInfo['fileInfo']
        file_hash = file_info['fileHash']
        if not b64decode(file_hash) == pyblake2.blake2s(self.chunks_block).digest():
            raise ValueError("ciphertext does not match given hash!")
        symbox = SymmetricMiniLock.from_key(b64decode(file_info['fileKey']))
        filename, *filechunks = symbox.decrypt(self.chunks_block, b64decode(file_info['fileNonce']))
        try:
            filename = filename.decode('utf8')
        except Exception as E:
            raise ValueError("Cannot decode filename to UTF8 string: '{}'".format(filename))
        sender = decryptInfo['senderID']
        return filename, sender, b''.join(filechunks)