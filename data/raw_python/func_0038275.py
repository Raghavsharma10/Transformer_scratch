def get_group_filters(self):
        """Return es OR filters to include all special coverage group conditions."""
        group_filters = []
        field_map = {
            "feature-type": "feature_type.slug",
            "tag": "tags.slug",
            "content-type": "_type"
        }
        for group_set in self.query.get("groups", []):
            for group in group_set:
                group_filter = es_filter.MatchAll()
                for condition in group.get("conditions", []):
                    group_filter &= get_condition_filter(condition, field_map=field_map)
                group_filters.append(group_filter)
        return group_filters