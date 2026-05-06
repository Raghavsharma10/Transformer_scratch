def _build_hash_secret(self, seq_seed, seq_len=HASH_SECRET_LENGTH,
            mod_value=HASH_SECRET_MOD_CONST):
        """Build a seed for the hash based on the Fibonacci sequence

        Take first `seq_len` + len(`seq_seed`) characters of Fibonacci
        sequence, starting with `seq_seed`, and applying e % `mod_value` +
        `HASH_SECRET_CHAR_OFFSET` to the resulting sequence, then return as
        a string

        @param tuple|list seq_seed
        @param int seq_len
        @param int mod_value
        @return str
        """

        # make sure we use a list, tuples are immutable
        fbn_seq = list(seq_seed)
        for i in range(seq_len):
            fbn_seq.append(fbn_seq[-1] + fbn_seq[-2])
        hash_secret = list(map(
            lambda c: chr(c % mod_value + self.HASH_SECRET_CHAR_OFFSET),
            fbn_seq[2:]))
        return ''.join(hash_secret)