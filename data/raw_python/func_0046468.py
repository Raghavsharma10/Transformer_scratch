def cache_from_names(self):
        """Yield the image names to do --cache-from from"""
        cache_from = self.cache_from()

        if not cache_from or cache_from is NotSpecified:
            return

        if cache_from is True:
            yield self.image_name
            return

        for thing in cache_from:
            if not isinstance(thing, six.string_types):
                yield thing.image_name
            else:
                yield thing