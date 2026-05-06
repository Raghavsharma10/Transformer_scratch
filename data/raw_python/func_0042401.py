def leaves(self, name):
        """
        RETURN LEAVES OF GIVEN PATH NAME
        pull leaves, considering query_path and namespace
        pull all first-level properties
        pull leaves, including parent leaves
        pull the head of any tree by name
        :param name:
        :return:
        """

        return list(self.lookup_leaves.get(unnest_path(name), Null))