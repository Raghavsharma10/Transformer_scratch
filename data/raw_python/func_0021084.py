def is_sub_to_all_kinds(self, *super_entity_kinds):
        """
        Each returned entity will have superentites whos combined entity_kinds included *super_entity_kinds
        """
        if super_entity_kinds:
            if len(super_entity_kinds) == 1:
                # Optimize for the case of just one
                has_subset = EntityRelationship.objects.filter(
                    super_entity__entity_kind=super_entity_kinds[0]).values_list('sub_entity', flat=True)
            else:
                # Get a list of entities that have super entities with all types
                has_subset = EntityRelationship.objects.filter(
                    super_entity__entity_kind__in=super_entity_kinds).values('sub_entity').annotate(
                    Count('super_entity')).filter(super_entity__count=len(set(super_entity_kinds))).values_list(
                    'sub_entity', flat=True)

            return self.filter(pk__in=has_subset)
        else:
            return self