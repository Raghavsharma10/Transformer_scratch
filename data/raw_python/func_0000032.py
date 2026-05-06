def mul_block(self, index, val):
        """Multiply values in block"""
        self._prepare_cache_slice(index)
        self.msinds[self.cache_slice] *= val