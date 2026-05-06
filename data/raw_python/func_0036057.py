def gen_challenge(self, state):
        """returns the next challenge and increments the seed and index
        in the state.

        :param state: the state to use for generating the challenge.  will
        verify the integrity of the state object before using it to generate
        a challenge.  it will then modify the state by incrementing the seed
        and index and resign the state for passing back to the server for
        storage
        """
        state.checksig(self.key)
        if (state.index >= state.n):
            raise HeartbeatError("Out of challenges.")
        state.seed = MerkleHelper.get_next_seed(self.key, state.seed)
        chal = Challenge(state.seed, state.index)
        state.index += 1
        state.sign(self.key)
        return chal