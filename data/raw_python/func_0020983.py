def upsert_entity_kinds(self, entity_kinds):
        """
        Given a list of entity kinds ensure they are synced properly to the database.
        This will ensure that only unchanged entity kinds are synced and will still return all
        updated entity kinds

        :param entity_kinds: The list of entity kinds to sync
        """

        # Filter out unchanged entity kinds
        unchanged_entity_kinds = {}
        if entity_kinds:
            unchanged_entity_kinds = {
                (entity_kind.name, entity_kind.display_name): entity_kind
                for entity_kind in EntityKind.all_objects.extra(
                    where=['(name, display_name) IN %s'],
                    params=[tuple(
                        (entity_kind.name, entity_kind.display_name)
                        for entity_kind in entity_kinds
                    )]
                )
            }

        # Filter out the unchanged entity kinds
        changed_entity_kinds = [
            entity_kind
            for entity_kind in entity_kinds
            if (entity_kind.name, entity_kind.display_name) not in unchanged_entity_kinds
        ]

        # If any of our kinds have changed upsert them
        upserted_enitity_kinds = []
        if changed_entity_kinds:
            # Select all our existing entity kinds for update so we can do proper locking
            # We have to select all here for some odd reason, if we only select the ones
            # we are syncing we still run into deadlock issues
            list(EntityKind.all_objects.all().select_for_update().values_list('id', flat=True))

            # Upsert the entity kinds
            upserted_enitity_kinds = manager_utils.bulk_upsert(
                queryset=EntityKind.all_objects.filter(
                    name__in=[entity_kind.name for entity_kind in changed_entity_kinds]
                ),
                model_objs=changed_entity_kinds,
                unique_fields=['name'],
                update_fields=['display_name'],
                return_upserts=True
            )

        # Return all the entity kinds
        return upserted_enitity_kinds + list(unchanged_entity_kinds.values())