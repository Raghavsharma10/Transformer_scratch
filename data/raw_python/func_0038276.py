def query(self):
        """Group the self.special_coverages queries and memoize them."""
        if not self._query:
            self._query.update({
                "excluded_ids": [],
                "included_ids": [],
                "pinned_ids": [],
                "groups": [],
            })
            for special_coverage in self._special_coverages:
                # Access query at dict level.
                query = getattr(special_coverage, "query", {})
                if "query" in query:
                    query = query.get("query")
                self._query["excluded_ids"] += query.get("excluded_ids", [])
                self._query["included_ids"] += query.get("included_ids", [])
                self._query["pinned_ids"] += query.get("pinned_ids", [])
                self._query["groups"] += [query.get("groups", [])]
        return self._query