def is_sub_to_any(self, *super_entities):
        """
        Given a list of super entities, return the entities that have super entities that interset with those provided.
        """
        if super_entities:
            return self.filter(id__in=EntityRelationship.objects.filter(
                super_entity__in=super_entities).values_list('sub_entity', flat=True))
        else:
            return self