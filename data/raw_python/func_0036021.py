def encode(self, file):
        """This function returns a (tag,state) tuple that is calculated for
        the given file.  the state will be encrypted with `self.key`

        :param file: the file to encode
        """
        tag = Tag()
        tag.sigma = list()

        state = State(Random.new().read(32), Random.new().read(32))

        f = KeyedPRF(state.f_key, self.prime)
        alpha = KeyedPRF(state.alpha_key, self.prime)

        done = False
        chunk_id = 0

        while (not done):
            sigma = f.eval(chunk_id)
            for j in range(0, self.sectors):
                buffer = file.read(self.sectorsize)

                if (len(buffer) > 0):
                    sigma += alpha.eval(j) * number.bytes_to_long(buffer)

                if (len(buffer) != self.sectorsize):
                    done = True
                    break
            sigma %= self.prime
            tag.sigma.append(sigma)
            chunk_id += 1

        state.chunks = chunk_id
        state.encrypt(self.key)

        return (tag, state)