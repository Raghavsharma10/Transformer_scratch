def generate_challenges(self, num, root_seed):
        """ Generate the specified number of hash challenges.

        :param num: The number of hash challenges we want to generate.
        :param root_seed: Some value that we use to generate our seeds from.
        """

        # Generate a series of seeds
        seeds = self.generate_seeds(num, root_seed, self.secret)
        blocks = self.pick_blocks(num, root_seed)

        # List of 2-tuples (seed, hash_response)
        self.challenges = []

        # Generate the corresponding hash for each seed
        for i in range(num):
            self.challenges.append(Challenge(blocks[i], seeds[i]))
            response = self.meet_challenge(self.challenges[i])
            self.challenges[i].response = response