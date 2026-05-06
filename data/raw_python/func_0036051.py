def fromdict(dict):
        """Takes a dictionary as an argument and returns a new Tag object
        from the dictionary.

        :param dict: the dictionary to convert
        """
        tree = MerkleTree.fromdict(dict['tree'])
        chunksz = dict['chunksz']
        filesz = dict['filesz']
        return Tag(tree, chunksz, filesz)