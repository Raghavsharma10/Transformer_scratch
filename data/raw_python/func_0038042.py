def get_full_recirc_content(self, published=True):
        """performs es search and gets all content objects
        """
        q = self.get_query()
        search = custom_search_model(Content, q, published=published, field_map={
            "feature_type": "feature_type.slug",
            "tag": "tags.slug",
            "content-type": "_type"
        })
        return search