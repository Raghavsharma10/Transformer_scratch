def _categorize(self, slug, category):
        """Add the ``slug`` to the ``category``. We store category data as
        as set, with a key of the form::

            c:<category name>

        The data is set of metric slugs::

            "slug-a", "slug-b", ...

        """
        key = self._category_key(category)
        self.r.sadd(key, slug)

        # Store all category names in a Redis set, for easy retrieval
        self.r.sadd(self._categories_key, category)