def verify_branch(leaf, branch, root):
        """This will verify that the given branch fits the given leaf and root
        It calculates the hash of the leaf, and then verifies that one of the
        bottom level nodes in the branch matches the leaf hash.  Then it
        calculates the hash of the two nodes on the next level and checks that
        one of the nodes on the level above matches.  It continues this until
        it reaches the top level of the tree where it asserts that the root is
        equal to the hash of the nodes below

        :param leaf: the leaf to check
        :param branch: a list of tuples (pairs) of the nodes in the branch,
        ordered from leaf to root.
        :param root: the root node
        """
        # just check the hashes are correct
        try:
            lh = leaf.get_hash()
        except:
            return False
        for i in range(0, branch.get_order()):
            if (branch.get_left(i) != lh and branch.get_right(i) != lh):
                return False
            h = hashlib.sha256()
            if (len(branch.get_left(i)) > 0):
                h.update(branch.get_left(i))
            if (len(branch.get_right(i)) > 0):
                h.update(branch.get_right(i))
            lh = h.digest()
        if (root != lh):
            return False
        return True