def field_pklist_from_json(self, data):
        """Load a PkOnlyQueryset from a JSON dict.

        This uses the same format as cached_queryset_from_json
        """
        model = get_model(data['app'], data['model'])
        return PkOnlyQueryset(self, model, data['pks'])