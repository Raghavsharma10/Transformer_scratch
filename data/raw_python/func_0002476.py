def with_(self, *relations):
        """
        Set the relationships that should be eager loaded.

        :return: The current Builder instance
        :rtype: Builder
        """
        if not relations:
            return self

        eagers = self._parse_relations(list(relations))

        self._eager_load.update(eagers)

        return self