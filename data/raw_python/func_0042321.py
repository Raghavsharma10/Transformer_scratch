def from_passphrase(cls, email, passphrase):
        """
        This performs key derivation from an email address and passphrase according
        to the miniLock specification.
        
        Specifically, the passphrase is digested with a standard blake2s 32-bit digest,
        then it is passed through scrypt with the email address as salt value using
        N = 217, r = 8, p = 1, L = 32.
        
        The 32-byte digest from scrypt is then used as the Private Key from which
        the public key is derived.
        """
        pp_blake = pyblake2.blake2s(cls.ensure_bytes(passphrase)).digest()
        #pp_scrypt = scrypt.hash(pp_blake, cls.ensure_bytes(email), 2**17, 8, 1, 32)
        pp_scrypt = pylibscrypt.scrypt(pp_blake, cls.ensure_bytes(email), 2**17, 8, 1, 32)
        key = nacl.public.PrivateKey(pp_scrypt)
        return cls(key.public_key, key)