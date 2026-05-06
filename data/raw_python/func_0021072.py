def reverse(self):
        """
        Reverses the items of this collection "in place" (only two values are
        retrieved from Redis at a time).
        """
        def reverse_trans(pipe):
            if self.writeback:
                self._sync_helper(pipe)

            n = self.__len__(pipe)
            for i in range(n // 2):
                left = pipe.lindex(self.key, i)
                right = pipe.lindex(self.key, n - i - 1)
                pipe.lset(self.key, i, right)
                pipe.lset(self.key, n - i - 1, left)

        self._transaction(reverse_trans)