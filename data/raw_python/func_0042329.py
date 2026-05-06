def new(cls, file_name, file_or_contents, sender, recipients):
        """
        Constructs (that is, encrypts) a new miniLock file from sender to recipients.
        """
        assert_type_and_length('recipients', recipients, list, minL=1)
        assert_type_and_length('sender', sender, UserLock)
        for R in recipients:
            assert_type_and_length('recipient', R, (str, UserLock))   
        recipients = list(set(recipients))
        # Encrypt file with secret key using file_contents and file_name
        file_key   = os.urandom(32)
        file_nonce = os.urandom(16)
        file_cipher = SymmetricMiniLock.from_key(file_key)
        ciphertext = b''.join(file_cipher.encrypt(file_or_contents, file_name, file_nonce))
        file_info = {
            'fileKey'   : b64encode(file_key),
            'fileNonce' : b64encode(file_nonce),
            'fileHash'  : b64encode(pyblake2.blake2s(ciphertext).digest())
        }
        header = MiniLockHeader.new(file_info, sender, recipients)
        b_header = header.to_bytes()
        encrypted_file = b'miniLock' + len(b_header).to_bytes(4, 'little') + b_header + ciphertext
        return cls(encrypted_file)