def full_outer(self):
        """
            Performs Full Outer Join
            :return full_outer: dict
        """
        self.get_collections_data()
        full_outer_join = self.merge_join_docs(
            set(self.collections_data['left'].keys()) | set(self.collections_data['right'].keys()))
        return full_outer_join