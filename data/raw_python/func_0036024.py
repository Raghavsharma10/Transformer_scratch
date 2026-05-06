def verify(self, proof, chal, state):
        """This returns True if the proof matches the challenge and file state

        :param proof: the proof that was returned from the server
        :param chal: the challenge sent to the server
        :param state: the state of the file, which can be encrypted
        """
        state.decrypt(self.key)

        index = KeyedPRF(chal.key, state.chunks)
        v = KeyedPRF(chal.key, chal.v_max)
        f = KeyedPRF(state.f_key, self.prime)
        alpha = KeyedPRF(state.alpha_key, self.prime)

        rhs = 0

        for i in range(0, chal.chunks):
            rhs += v.eval(i) * f.eval(index.eval(i))

        for j in range(0, self.sectors):
            rhs += alpha.eval(j) * proof.mu[j]

        rhs %= self.prime
        return proof.sigma == rhs