def remove_entity(self, entity, sub_entity_kind=None):
        """
        Remove an entity, or sub-entity group to this EntityGroup.

        :type entity: Entity
        :param entity: The entity to remove.

        :type sub_entity_kind: Optional EntityKind
        :param sub_entity_kind: If a sub_entity_kind is given, all
            sub_entities of the entity will be removed from this
            EntityGroup.
        """
        EntityGroupMembership.objects.get(
            entity_group=self,
            entity=entity,
            sub_entity_kind=sub_entity_kind,
        ).delete()