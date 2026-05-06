def meet_challenge(self, challenge):
        """ Get the SHA256 hash of a specific file block plus the provided
        seed. The default block size is one tenth of the file. If the file is
        larger than 10KB, 1KB is used as the block size.

        :param challenge: challenge as a `Challenge <heartbeat.Challenge>`
        object
        """
        chunk_size = min(1024, self.file_size // 10)
        seed = challenge.seed

        h = hashlib.sha256()
        self.file_object.seek(challenge.block)

        if challenge.block > (self.file_size - chunk_size):
            end_slice = (
                challenge.block - (self.file_size - chunk_size)
            )
            h.update(self.file_object.read(end_slice))
            self.file_object.seek(0)
            h.update(self.file_object.read(chunk_size - end_slice))
        else:
            h.update(self.file_object.read(chunk_size))

        h.update(seed)

        return h.digest()