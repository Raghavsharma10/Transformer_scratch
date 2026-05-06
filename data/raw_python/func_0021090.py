def get_membership_cache(self, group_ids=None, is_active=True):
        """
        Build a dict cache with the group membership info. Keyed off the group id and the values are
        a 2 element list of entity id and entity kind id (same values as the membership model). If no group ids
        are passed, then all groups will be fetched

        :param is_active: Flag indicating whether to filter on entity active status. None will not filter.
        :rtype: dict
        """
        membership_queryset = EntityGroupMembership.objects.filter(
            Q(entity__isnull=True) | (Q(entity__isnull=False) & Q(entity__is_active=is_active))
        )
        if is_active is None:
            membership_queryset = EntityGroupMembership.objects.all()

        if group_ids:
            membership_queryset = membership_queryset.filter(entity_group_id__in=group_ids)

        membership_queryset = membership_queryset.values_list('entity_group_id', 'entity_id', 'sub_entity_kind_id')

        # Iterate over the query results and build the cache dict
        membership_cache = {}
        for entity_group_id, entity_id, sub_entity_kind_id in membership_queryset:
            membership_cache.setdefault(entity_group_id, [])
            membership_cache[entity_group_id].append([entity_id, sub_entity_kind_id])

        return membership_cache