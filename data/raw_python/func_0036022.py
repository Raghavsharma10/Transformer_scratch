def gen_challenge(self, state):
        """This function generates a challenge for given state.  It selects a
        random number and sets that as the challenge key.  By default, v_max
        is set to the prime, and the number of chunks to challenge is the
        number of chunks in the file.  (this doesn't guarantee that the whole
        file will be checked since some chunks could be selected twice and
        some selected none.

        :param state: the state to use.  it can be encrypted, as it will
        have just been received from the server
        """
        state.decrypt(self.key)

        chal = Challenge(state.chunks, self.prime, Random.new().read(32))

        return chal