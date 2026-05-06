def bulk_overwrite(self, entities_and_kinds):
        """
        Update the group to the given entities and sub-entity groups.

        After this operation, the only members of this EntityGroup
        will be the given entities, and sub-entity groups.

        :type entities_and_kinds: List of (Entity, EntityKind) pairs.
        :param entities_and_kinds: A list of entity, entity-kind pairs
            to set to the EntityGroup. In the pairs the entity-kind
            can be ``None``, to add a single entity, or some entity
            kind to add all sub-entities of that kind.
        """
        EntityGroupMembership.objects.filter(entity_group=self).delete()
        return self.bulk_add_entities(entities_and_kinds)