def search(self):
        """Return a search using the combined query of all associated special coverage objects."""
        # Retrieve all Or filters pertinent to the special coverage query.
        should_filters = [
            es_filter.Terms(pk=self.query.get("included_ids", [])),
            es_filter.Terms(pk=self.query.get("pinned_ids", []))
        ]
        should_filters += self.get_group_filters()

        # Compile list of all Must filters.
        must_filters = [
            es_filter.Bool(should=should_filters),
            ~es_filter.Terms(pk=self.query.get("excluded_ids", []))
        ]

        return Content.search_objects.search().filter(es_filter.Bool(must=must_filters))