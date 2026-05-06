def upsert_entity_relationships(self, queryset, entity_relationships):
        """
        Upsert entity relationships to the database
        :param queryset: The base queryset to use
        :param entity_relationships: The entity relationships to ensure exist in the database
        """

        # Select the relationships for update
        if entity_relationships:
            list(queryset.select_for_update().values_list(
                'id',
                flat=True
            ))

        # Sync the relationships
        return manager_utils.sync(
            queryset=queryset,
            model_objs=entity_relationships,
            unique_fields=['sub_entity_id', 'super_entity_id'],
            update_fields=[],
            return_upserts=True
        )