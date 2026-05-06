def pick_blocks(self, num, root_seed):
        """ Pick a set of positions to start reading blocks from the file
        that challenges are created for. This is a deterministic
        operation. Positions are guaranteed to be within the bounds of the
        file.

        :param num: Number of blocks to pick
        :param root_seed: Seed with which begin picking blocks.
        :return: block values as a list
        """
        if num < 0:
            raise HeartbeatError('%s is not greater than 0' % num)

        blocks = []
        random.seed(root_seed)

        for i in range(num):
            blocks.append(random.randint(0, self.file_size - 1))

        return blocks