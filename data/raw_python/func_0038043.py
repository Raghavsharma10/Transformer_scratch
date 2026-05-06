def has_pinned_content(self):
        """determines if the there is a pinned object in the search
        """
        q = self.get_query()
        if "pinned_ids" in q:
            return bool(len(q.get("pinned_ids", [])))
        return False