def prove(self, file, chal, tag):
        """This function returns a proof calculated from the file, the
        challenge, and the file tag

        :param file: this is a file like object that supports `read()`,
        `tell()` and `seek()` methods.
        :param chal: the challenge to use for proving
        :param tag: the file tag
        """
        chunk_size = self.sectors * self.sectorsize

        index = KeyedPRF(chal.key, len(tag.sigma))
        v = KeyedPRF(chal.key, chal.v_max)

        proof = Proof()
        proof.mu = [0] * self.sectors
        proof.sigma = 0

        for i in range(0, chal.chunks):
            for j in range(0, self.sectors):
                pos = index.eval(i) * chunk_size + j * self.sectorsize
                file.seek(pos)
                buffer = file.read(self.sectorsize)
                if (len(buffer) > 0):
                    proof.mu[j] += v.eval(i) * number.bytes_to_long(buffer)

                if (len(buffer) != self.sectorsize):
                    break

        for j in range(0, self.sectors):
            proof.mu[j] %= self.prime

        for i in range(0, chal.chunks):
            proof.sigma += v.eval(i) * tag.sigma[index.eval(i)]

        proof.sigma %= self.prime

        return proof