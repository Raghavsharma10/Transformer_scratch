def bulk_remove_entities(self, entities_and_kinds):
        """
        Remove many entities and sub-entity groups to this EntityGroup.

        :type entities_and_kinds: List of (Entity, EntityKind) pairs.
        :param entities_and_kinds: A list of entity, entity-kind pairs
            to remove from the group. In the pairs, the entity-kind
            can be ``None``, to add a single entity, or some entity
            kind to add all sub-entities of that kind.
        """
        criteria = [
            Q(entity=entity, sub_entity_kind=entity_kind)
            for entity, entity_kind in entities_and_kinds
        ]
        criteria = reduce(lambda q1, q2: q1 | q2, criteria, Q())
        EntityGroupMembership.objects.filter(
            criteria, entity_group=self).delete()