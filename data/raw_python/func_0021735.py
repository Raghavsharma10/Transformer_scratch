def _build_encryption_key(self, subtitle_id, key_size=ENCRYPTION_KEY_SIZE):
        """Generate the encryption key for a given media item

        Encryption key is basically just
        sha1(<magic value based on subtitle_id> + '"#$&).6CXzPHw=2N_+isZK') then
        padded with 0s to 32 chars

        @param int subtitle_id
        @param int key_size
        @return str
        """

        # generate a 160-bit SHA1 hash
        sha1_hash = hashlib.new('sha1', self._build_hash_secret((1, 2)) +
            self._build_hash_magic(subtitle_id)).digest()
        # pad to 256-bit hash for 32 byte key
        sha1_hash += '\x00' * max(key_size - len(sha1_hash), 0)
        return sha1_hash[:key_size]