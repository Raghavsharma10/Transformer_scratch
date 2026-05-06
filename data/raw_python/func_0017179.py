def remove_sister(self, sister=None):
        """
        Removes a sister node. It has the same effect as
        **`TreeNode.up.remove_child(sister)`**

        If a sister node is not supplied, the first sister will be deleted
        and returned.

        :argument sister: A node instance

        :return: The node removed
        """
        sisters = self.get_sisters()
        if len(sisters) > 0:
            if sister is None:
                sister = sisters.pop(0)
            return self.up.remove_child(sister)