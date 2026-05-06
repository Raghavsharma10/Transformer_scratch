def get_all_entities(self, membership_cache=None, entities_by_kind=None, return_models=False, is_active=True):
        """
        Returns a list of all entity ids in this group or optionally returns a queryset for all entity models.
        In order to reduce queries for multiple group lookups, it is expected that the membership_cache and
        entities_by_kind are built outside of this method and passed in as arguments.
        :param membership_cache: A group cache dict generated from `EntityGroup.objects.get_membership_cache()`
        :type membership_cache: dict
        :param entities_by_kind: An entities by kind dict generated from the `get_entities_by_kind` function
        :type entities_by_kind: dict
        :param return_models: If True, returns an Entity queryset, if False, returns a set of entity ids
        :type return_models: bool
        :param is_active: Flag to control entities being returned. Defaults to True for active entities only
        :type is_active: bool
        """
        # If cache args were not passed, generate the cache
        if membership_cache is None:
            membership_cache = EntityGroup.objects.get_membership_cache([self.id], is_active=is_active)

        if entities_by_kind is None:
            entities_by_kind = entities_by_kind or get_entities_by_kind(membership_cache=membership_cache)

        # Build set of all entity ids for this group
        entity_ids = set()

        # This group does have entities
        if membership_cache.get(self.id):

            # Loop over each membership in this group
            for entity_id, entity_kind_id in membership_cache[self.id]:
                if entity_id:
                    if entity_kind_id:
                        # All sub entities of this kind under this entity
                        entity_ids.update(entities_by_kind[entity_kind_id][entity_id])
                    else:
                        # Individual entity
                        entity_ids.add(entity_id)
                else:
                    # All entities of this kind
                    entity_ids.update(entities_by_kind[entity_kind_id]['all'])

        # Check if a queryset needs to be returned
        if return_models:
            return Entity.objects.filter(id__in=entity_ids)

        return entity_ids