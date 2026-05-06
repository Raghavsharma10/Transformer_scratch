def ephemeral(cls):
        """
        Creates a new ephemeral key constructed using a raw 32-byte string from urandom.
        Ephemeral keys are used once for each encryption task and are then discarded;
        they are not intended for long-term or repeat use.
        """
        private_key = nacl.public.PrivateKey(os.urandom(32))
        return cls(private_key.public_key, private_key)