def generate(bits, progress_func=None):
        """
        Generate a new private RSA key.  This factory function can be used to
        generate a new host key or authentication key.

        @param bits: number of bits the generated key should be.
        @type bits: int
        @param progress_func: an optional function to call at key points in
            key generation (used by C{pyCrypto.PublicKey}).
        @type progress_func: function
        @return: new private key
        @rtype: L{RSAKey}
        """
        rsa = RSA.generate(bits, rng.read, progress_func)
        key = RSAKey(vals=(rsa.e, rsa.n))
        key.d = rsa.d
        key.p = rsa.p
        key.q = rsa.q
        return key