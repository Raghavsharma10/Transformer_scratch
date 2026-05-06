def _build_hash_magic(self, subtitle_id):
        """Build the other half of the encryption key hash

        I have no idea what is going on here

        @param int subtitle_id
        @return str
        """

        media_magic = self.HASH_MAGIC_CONST ^ subtitle_id
        hash_magic = media_magic ^ media_magic >> 3 ^ media_magic * 32
        return str(hash_magic)