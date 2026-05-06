def is_contains(self, data):
        """
            Judge the data whether is already exist if each bit of hash code is 1 then data exist.
        """
        if not data:
            return False
        data = self._compress_by_md5(data)
        result = True
        # cut the first two place,route to different block by block_num
        name = self.key + str(int(data[0:2], 16) % self.block_num)
        for h in self.hash_function:
            local_hash = h.hash(data)
            result = result & self.server.getbit(name, local_hash)
        return result