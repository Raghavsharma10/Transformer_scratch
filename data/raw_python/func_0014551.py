def generate(bits=1024, progress_func=None):
        """
        Generate a new private DSS key.  This factory function can be used to
        generate a new host key or authentication key.

        @param bits: number of bits the generated key should be.
        @type bits: int
        @param progress_func: an optional function to call at key points in
            key generation (used by C{pyCrypto.PublicKey}).
        @type progress_func: function
        @return: new private key
        @rtype: L{DSSKey}
        """
        dsa = DSA.generate(bits, rng.read, progress_func)
        key = DSSKey(vals=(dsa.p, dsa.q, dsa.g, dsa.y))
        key.x = dsa.x
        return key