def upsert_entities(self, entities, sync=False):
        """
        Upsert a list of entities to the database
        :param entities: The entities to sync
        :param sync: Do a sync instead of an upsert
        """

        # Select the entities we are upserting for update to reduce deadlocks
        if entities:
            # Default select for update query when syncing all
            select_for_update_query = (
                'SELECT FROM {table_name} FOR NO KEY UPDATE'
            ).format(
                table_name=Entity._meta.db_table
            )
            select_for_update_query_params = []

            # If we are not syncing all, only select those we are updating
            if not sync:
                select_for_update_query = (
                    'SELECT FROM {table_name} WHERE (entity_type_id, entity_id) IN %s FOR NO KEY UPDATE'
                ).format(
                    table_name=Entity._meta.db_table
                )
                select_for_update_query_params = [tuple(
                    (entity.entity_type_id, entity.entity_id)
                    for entity in entities
                )]

            # Select the items for update
            with connection.cursor() as cursor:
                cursor.execute(select_for_update_query, select_for_update_query_params)

        # If we are syncing run the sync logic
        if sync:
            upserted_entities = manager_utils.sync(
                queryset=Entity.all_objects.all(),
                model_objs=entities,
                unique_fields=['entity_type_id', 'entity_id'],
                update_fields=['entity_kind_id', 'entity_meta', 'display_name', 'is_active'],
                return_upserts=True
            )
        # Otherwise we want to upsert our entities
        else:
            upserted_entities = manager_utils.bulk_upsert(
                queryset=Entity.all_objects.extra(
                    where=['(entity_type_id, entity_id) IN %s'],
                    params=[tuple(
                        (entity.entity_type_id, entity.entity_id)
                        for entity in entities
                    )]
                ),
                model_objs=entities,
                unique_fields=['entity_type_id', 'entity_id'],
                update_fields=['entity_kind_id', 'entity_meta', 'display_name', 'is_active'],
                return_upserts=True
            )

        # Return the upserted entities
        return upserted_entities