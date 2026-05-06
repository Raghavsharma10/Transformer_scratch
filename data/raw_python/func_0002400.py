def load(self, *relations):
        """
        Load a set of relationships onto the collection.
        """
        if len(self._items) > 0:
            query = self.first().new_query().with_(*relations)

            self._items = query.eager_load_relations(self._items)

        return self