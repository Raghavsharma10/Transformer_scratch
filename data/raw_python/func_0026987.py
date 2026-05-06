def to_representation(self, obj):
        """
        Serializes any object to his url representation
        """
        kwargs = None
        for field in self.lookup_fields:
            if hasattr(obj, field):
                kwargs = {field: getattr(obj, field)}
                break
        if kwargs is None:
            raise AttributeError('Related object does not have any of lookup_fields')
        request = self._get_request()
        return request.build_absolute_uri(reverse(self._get_url(obj), kwargs=kwargs))