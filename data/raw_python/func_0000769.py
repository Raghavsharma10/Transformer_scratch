def _location(self, obj):
        """ Get location of the `obj`

        Arguments:
            :obj: self.Model instance.
        """
        field_name = self.clean_id_name
        return self.request.route_url(
            self._resource.uid,
            **{self._resource.id_name: getattr(obj, field_name)})