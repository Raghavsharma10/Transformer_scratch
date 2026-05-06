def set_attributes(self, **kwargs):
        """
        Similar to BaseResource.set_attributes except
        it will attempt to match URL strings with registered
        related resources, and build their get_* method and
        attach it to this resource.
        """
        if not self.Meta.related_resources:
            # Just do what the normal BaseResource does
            super(HypermediaResource, self).set_attributes(**kwargs)
            return

        # Extract all the values that are URLs
        url_values = {}
        for k, v in kwargs.items():
            try:
                if isinstance(v, list):
                    [self._parse_url_and_validate(i) for i in v]
                else:
                    self._parse_url_and_validate(v)
                url_values[k] = v
            except BadURLException:
                # This is a badly formed URL or not a URL at all, so skip
                pass
        # Assign the valid method values and then remove them from the kwargs
        assigned_values = self.match_urls_to_resources(url_values)
        for k in assigned_values.keys():
            kwargs.pop(k, None)
        # Assign the rest as attributes.
        for field, value in kwargs.items():
            if field in self.Meta.attributes:
                setattr(self, field, value)