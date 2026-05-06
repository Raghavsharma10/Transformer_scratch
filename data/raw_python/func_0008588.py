def block_by_command(cls, command):
        """Return the block with the given :attr:`command`.

        Returns None if the block is not found.

        """
        for block in cls.blocks:
            if block.has_command(command):
                return block