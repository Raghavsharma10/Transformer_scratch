def inner(self):
        """
            Performs Inner Join
            :return inner_join: dict
        """
        self.get_collections_data()

        inner_join = self.merge_join_docs(set(self.collections_data['left'].keys()) & set(
            self.collections_data['right'].keys()))

        return inner_join