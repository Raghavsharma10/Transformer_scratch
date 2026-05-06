def filter_by_name_or_id_or_tag(self, query_string, queryset = None):
        """Find objects that match the identifier of form {name}@{ID}, {name},
        or @{ID}, where ID may be truncated
        """
        assert self.Model.NAME_FIELD, \
            'NAME_FIELD is missing on model %s' % self.Model.__name__
        assert self.Model.ID_FIELD, \
            'ID_FIELD is missing on model %s' % self.Model.__name__
        assert self.Model.TAG_FIELD, \
            'TAG_FIELD is missing on model %s' % self.Model.__name__

        filter_args = {}
        name, uuid, tag = self._parse_as_name_or_id_or_tag(query_string)
        if name is not None:
            filter_args[self.Model.NAME_FIELD] = name
        if uuid is not None:
            filter_args[self.Model.ID_FIELD+'__startswith'] = uuid
        if tag is not None:
            filter_args[self.Model.TAG_FIELD] = tag
        if queryset is None:
            queryset = self.Model.objects.all()
        return queryset.filter(**filter_args)