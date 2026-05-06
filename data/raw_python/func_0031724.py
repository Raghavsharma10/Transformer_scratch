def right_outer(self):
        """
            Performs Right Outer Join
            :return right_outer: dict
        """
        self.get_collections_data()
        right_outer_join = self.merge_join_docs(
            set(self.collections_data['right'].keys()))
        return right_outer_join