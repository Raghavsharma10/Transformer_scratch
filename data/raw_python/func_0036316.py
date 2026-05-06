def insert(self, data):
        """
            Insert 1 into each bit by local_hash
        """
        if not data:
            return
        data = self._compress_by_md5(data)
        # cut the first two place,route to different block by block_num
        name = self.key + str(int(data[0:2], 16) % self.block_num)
        for h in self.hash_function:
            local_hash = h.hash(data)
            self.server.setbit(name, local_hash, 1)