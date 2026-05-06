def is_sub_to_any_kind(self, *super_entity_kinds):
        """
        Find all entities that have super_entities of any of the specified kinds
        """
        if super_entity_kinds:
            # get the pks of the desired subs from the relationships table
            if len(super_entity_kinds) == 1:
                entity_pks = EntityRelationship.objects.filter(
                    super_entity__entity_kind=super_entity_kinds[0]
                ).select_related('entity_kind', 'sub_entity').values_list('sub_entity', flat=True)
            else:
                entity_pks = EntityRelationship.objects.filter(
                    super_entity__entity_kind__in=super_entity_kinds
                ).select_related('entity_kind', 'sub_entity').values_list('sub_entity', flat=True)
            # return a queryset limited to only those pks
            return self.filter(pk__in=entity_pks)
        else:
            return self