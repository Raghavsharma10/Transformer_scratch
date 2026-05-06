def get_entities_by_kind(membership_cache=None, is_active=True):
    """
    Builds a dict with keys of entity kinds if and values are another dict. Each of these dicts are keyed
    off of a super entity id and optional have an 'all' key for any group that has a null super entity.
    Example structure:
    {
        entity_kind_id: {
            entity1_id: [1, 2, 3],
            entity2_id: [4, 5, 6],
            'all': [1, 2, 3, 4, 5, 6]
        }
    }

    :rtype: dict
    """
    # Accept an existing cache or build a new one
    if membership_cache is None:
        membership_cache = EntityGroup.objects.get_membership_cache(is_active=is_active)

    entities_by_kind = {}
    kinds_with_all = set()
    kinds_with_supers = set()
    super_ids = set()

    # Loop over each group
    for group_id, memberships in membership_cache.items():

        # Look at each membership
        for entity_id, entity_kind_id in memberships:

            # Only care about memberships with entity kind
            if entity_kind_id:

                # Make sure a dict exists for this kind
                entities_by_kind.setdefault(entity_kind_id, {})

                # Check if this is all entities of a kind under a specific entity
                if entity_id:
                    entities_by_kind[entity_kind_id][entity_id] = []
                    kinds_with_supers.add(entity_kind_id)
                    super_ids.add(entity_id)
                else:
                    # This is all entities of this kind
                    entities_by_kind[entity_kind_id]['all'] = []
                    kinds_with_all.add(entity_kind_id)

    # Get entities for 'all'
    all_entities_for_types = Entity.objects.filter(
        entity_kind_id__in=kinds_with_all
    ).values_list('id', 'entity_kind_id')

    # Add entity ids to entity kind's all list
    for id, entity_kind_id in all_entities_for_types:
        entities_by_kind[entity_kind_id]['all'].append(id)

    # Get relationships
    relationships = EntityRelationship.objects.filter(
        super_entity_id__in=super_ids,
        sub_entity__entity_kind_id__in=kinds_with_supers
    ).values_list(
        'super_entity_id', 'sub_entity_id', 'sub_entity__entity_kind_id'
    )

    # Add entity ids to each super entity's list
    for super_entity_id, sub_entity_id, sub_entity__entity_kind_id in relationships:
        entities_by_kind[sub_entity__entity_kind_id].setdefault(super_entity_id, [])
        entities_by_kind[sub_entity__entity_kind_id][super_entity_id].append(sub_entity_id)

    return entities_by_kind