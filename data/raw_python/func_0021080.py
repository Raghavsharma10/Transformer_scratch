def rotate(self, n=1):
        """
        Rotate the deque n steps to the right.
        If n is negative, rotate to the left.
        """
        # No work to do for a 0-step rotate
        if n == 0:
            return

        def rotate_trans(pipe):
            # Synchronize the cache before rotating
            if self.writeback:
                self._sync_helper(pipe)

            # Rotating len(self) times has no effect.
            len_self = self.__len__(pipe)
            steps = abs_n % len_self

            # When n is positive we can use the built-in Redis command
            if forward:
                pipe.multi()
                for __ in range(steps):
                    pipe.rpoplpush(self.key, self.key)
            # When n is negative we must use Python
            else:
                for __ in range(steps):
                    pickled_value = pipe.lpop(self.key)
                    pipe.rpush(self.key, pickled_value)

        forward = n >= 0
        abs_n = abs(n)
        self._transaction(rotate_trans)