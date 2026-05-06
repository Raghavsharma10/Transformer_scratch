def expand(self, key_array):
        """
            Expand the encryption key per AES key schedule specifications

            http://en.wikipedia.org/wiki/Rijndael_key_schedule# Key_schedule_description
        """

        if len(key_array) != self._n:
            raise RuntimeError('expand(): key size ' + str(len(key_array)) + ' is invalid')

        # First n bytes are copied from key. Copy prevents inplace modification of original key
        new_key = list(key_array)

        rcon_iteration = 1
        len_new_key = len(new_key)

        # There are several parts of the code below that could be done with tidy list comprehensions like
        # the one I put in _core, but I left this alone for readability.

        # Grow the key until it is the correct length
        while len_new_key < self._b:

            # Copy last 4 bytes of extended key, apply _core function order i, increment i(rcon_iteration),
            # xor with 4 bytes n bytes from end of extended key
            t = new_key[-4:]
            t = self._core(t, rcon_iteration)
            rcon_iteration += 1
            t = self._xor_list(t, new_key[-self._n : -self._n + 4])# self._n_bytes_before(len_new_key, new_key))
            new_key.extend(t)
            len_new_key += 4

            # Run three passes of 4 byte expansion using copy of 4 byte tail of extended key
            # which is then xor'd with 4 bytes n bytes from end of extended key
            for j in range(3):
                t = new_key[-4:]
                t = self._xor_list(t, new_key[-self._n : -self._n + 4])
                new_key.extend(t)
                len_new_key += 4

            # If key length is 256 and key is not complete, add 4 bytes tail of extended key
            # run through sbox before xor with 4 bytes n bytes from end of extended key
            if self._key_length == 256 and len_new_key < self._b:
                t = new_key[-4:]
                t2=[]
                for x in t:
                    t2.append(aes_tables.sbox[x])
                t = self._xor_list(t2, new_key[-self._n : -self._n + 4])
                new_key.extend(t)
                len_new_key += 4

            # If key length is 192 or 256 and key is not complete, run 2 or 3 passes respectively
            # of 4 byte tail of extended key xor with 4 bytes n bytes from end of extended key
            if self._key_length != 128 and len_new_key < self._b:
                if self._key_length == 192:
                    r = range(2)
                else:
                    r = range(3)

                for j in r:
                    t = new_key[-4:]
                    t = self._xor_list(t, new_key[-self._n : -self._n + 4])
                    new_key.extend(t)
                    len_new_key += 4

        return new_key