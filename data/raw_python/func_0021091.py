def all_entities(self, is_active=True):
        """
        Return all the entities in the group.

        Because groups can contain both individual entities, as well
        as whole groups of entities, this method acts as a convenient
        way to get a queryset of all the entities in the group.
        """
        return self.get_all_entities(return_models=True, is_active=is_active)