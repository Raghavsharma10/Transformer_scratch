def get_validated_augment_query(self, augment_query=None):
        """
        Common rules for reading list augmentation hierarchy.

        1. Sponsored Content.
        2. Video Content.
        """
        augment_query = self.validate_query(augment_query)

        # Given an invalid query, reach for a Sponsored query.
        if not augment_query:
            augment_query = self.validate_query(Content.search_objects.sponsored())

        # Given an invalid Sponsored query, reach for a Video query.
        if not augment_query:
            reading_list_config = getattr(settings, "READING_LIST_CONFIG", {})
            excluded_channel_ids = reading_list_config.get("excluded_channel_ids", [])
            augment_query = self.validate_query(Content.search_objects.evergreen_video(
                excluded_channel_ids=excluded_channel_ids
            ))

        return augment_query