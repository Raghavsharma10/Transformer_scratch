def new(cls, file_info, sender, recipients, version=1):
        """
        Constructs (encrypts) a new MiniLockHeader object.
        file_info: dict, with 'fileKey', 'fileNonce', 'fileHash' as bytes entries
        sender: UserLock for sender
        recipients: list of UserLock for recipients
        """
        ephem_key = UserLock.ephemeral()
        decryptInfo = {}
        for recipient in recipients:
            if isinstance(recipient, str):
                recipient = UserLock.from_id(recipient)
            if not isinstance(recipient, UserLock):
                raise TypeError("Recipient must be either a UserLock object or a User ID as a string.")
            # This is the sender->recipient box for the inner fileinfo.
            sending_box = nacl.public.Box(sender.private_key, recipient.public_key)
            sending_nonce = os.urandom(24)
            sending_nonce_b64 = b64encode(sending_nonce)
            # Encrypt the fileinfo sender->recipient, then create an entry for this
            # recipient with senderID/recipientID.
            dumped_fileInfo = json.dumps(file_info, separators=(',',':')).encode('utf8')
            crypted_fileInfo = sending_box.encrypt(dumped_fileInfo, sending_nonce)[24:]
            di_entry = json.dumps({
                'fileInfo'    : b64encode(crypted_fileInfo),
                'senderID'    : sender.userID,
                'recipientID' : recipient.userID
            }).encode('utf8')
            # This is the ephem->recipient box, which obfuscates the sender.
            ephem_sending_box = nacl.public.Box(ephem_key.private_key, recipient.public_key)
            crypted_di_entry = ephem_sending_box.encrypt(di_entry, sending_nonce)[24:]
            decryptInfo[sending_nonce_b64] = b64encode(crypted_di_entry)
        # Now have a decryptInfo dictionary full of entries for each recipient.
        return cls({
            'version': version,
            'ephemeral': ephem_key.b64str,  # Should be ephem_key.userID, for consistency! Support both?
            'decryptInfo': decryptInfo
        })