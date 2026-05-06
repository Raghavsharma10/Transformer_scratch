def decrypt(self, recipient_key):
        """
        Attempt decryption of header with a private key; returns decryptInfo.
        Returns a dictionary, not a new MiniLockHeader!
        """
        ephem = UserLock.from_b64(self.dict['ephemeral'])
        ephem_box = nacl.public.Box(recipient_key.private_key, ephem.public_key)
        # Scan available entries in decryptInfo and try to decrypt each; break when
        # successful with any.
        for nonce, crypted_decryptInfo in self.dict['decryptInfo'].items():
            raw_nonce = b64decode(nonce)
            crypted_decryptInfo = b64decode(crypted_decryptInfo)
            try:
                decryptInfo_raw = ephem_box.decrypt(crypted_decryptInfo, raw_nonce)
                decryptInfo = json.loads(decryptInfo_raw.decode('utf8'))
                success_nonce = raw_nonce
                break
            except Exception as E:
                #print("Decoding exception: '{}' - with ciphertext '{}'".format(E, crypted_decryptInfo))
                pass
        else:
            raise ValueError("No decryptInfo block found for this recipient Key.")
        if not recipient_key.userID == decryptInfo['recipientID']:
            raise ValueError("Decrypted a meta block but stated recipient is not this private key!")
        # Now work with decryptInfo and success_nonce to extract file data.
        senderKey = UserLock.from_id(decryptInfo['senderID'])
        senderBox = nacl.public.Box(recipient_key.private_key, senderKey.public_key)
        fileInfo_raw = b64decode(decryptInfo['fileInfo'])
        fileInfo_decrypted = senderBox.decrypt(fileInfo_raw, success_nonce).decode('utf8')
        fileInfo = json.loads(fileInfo_decrypted)
        # Overwrite decryptInfo's fileInfo key
        decryptInfo['fileInfo'] = fileInfo
        return decryptInfo