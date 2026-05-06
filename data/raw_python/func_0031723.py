def left_outer(self):
        """
            Performs Left Outer Join
            :return left_outer: dict
        """
        self.get_collections_data()
        left_outer_join = self.merge_join_docs(
            set(self.collections_data['left'].keys()))
        return left_outer_join