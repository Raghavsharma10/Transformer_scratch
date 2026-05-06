def bulk_add_entities(self, entities_and_kinds):
        """
        Add many entities and sub-entity groups to this EntityGroup.

        :type entities_and_kinds: List of (Entity, EntityKind) pairs.
        :param entities_and_kinds: A list of entity, entity-kind pairs
            to add to the group. In the pairs the entity-kind can be
            ``None``, to add a single entity, or some entity kind to
            add all sub-entities of that kind.
        """
        memberships = [EntityGroupMembership(
            entity_group=self,
            entity=entity,
            sub_entity_kind=sub_entity_kind,
        ) for entity, sub_entity_kind in entities_and_kinds]
        created = EntityGroupMembership.objects.bulk_create(memberships)
        return created