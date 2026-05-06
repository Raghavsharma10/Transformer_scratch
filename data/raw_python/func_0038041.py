def get_recirc_content(self, published=True, count=3):
        """gets the first 3 content objects in the `included_ids`
        """
        query = self.get_query()

        # check if query has included_ids & if there are any ids in it,
        # in case the ids have been removed from the array
        if not query.get('included_ids'):
            qs = Content.search_objects.search()
            qs = qs.query(
                TagBoost(slugs=self.tags.values_list("slug", flat=True))
            ).filter(
                ~Ids(values=[self.id])
            ).sort(
                "_score"
            )
            return qs[:count]

        # NOTE: set included_ids to just be the first 3 ids,
        # otherwise search will return last 3 items
        query['included_ids'] = query['included_ids'][:count]
        search = custom_search_model(Content, query, published=published, field_map={
            "feature_type": "feature_type.slug",
            "tag": "tags.slug",
            "content-type": "_type"
        })
        return search