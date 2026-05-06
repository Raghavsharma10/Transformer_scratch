def augment_reading_list(self, primary_query, augment_query=None, reverse_negate=False):
        """Apply injected logic for slicing reading lists with additional content."""
        primary_query = self.validate_query(primary_query)
        augment_query = self.get_validated_augment_query(augment_query=augment_query)

        try:
            # We use this for cases like recent where queries are vague.
            if reverse_negate:
                primary_query = primary_query.filter(NegateQueryFilter(augment_query))
            else:
                augment_query = augment_query.filter(NegateQueryFilter(primary_query))
            augment_query = randomize_es(augment_query)
            return FirstSlotSlicer(primary_query, augment_query)
        except TransportError:
            return primary_query