def wait_for_n_keypresses(self, key, n=1):
        """Waits till one key was pressed n times.

        :param key: the key to be pressed as defined by pygame. E.g.
            pygame.K_LEFT for the left arrow key
        :type key: int
        :param n: number of repetitions till the function returns
        :type n: int
        """
        my_const = "key_consumed"
        counter = 0

        def keypress_listener(e): return my_const \
            if e.type == pygame.KEYDOWN and e.key == key \
            else EventConsumerInfo.DONT_CARE

        while counter < n:
            if self.listen(keypress_listener) == my_const:
                counter += 1