def get_content(self):
        """performs es search and gets content objects
        """
        if "query" in self.query:
            q = self.query["query"]
        else:
            q = self.query
        search = custom_search_model(Content, q, field_map={
            "feature-type": "feature_type.slug",
            "tag": "tags.slug",
            "content-type": "_type",
        })
        return search