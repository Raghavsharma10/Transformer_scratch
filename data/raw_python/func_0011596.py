def field_pklist_to_json(self, model, pks):
        """Convert a list of primary keys to a JSON dict.

        This uses the same format as cached_queryset_to_json
        """
        app_label = model._meta.app_label
        model_name = model._meta.model_name
        return {
            'app': app_label,
            'model': model_name,
            'pks': list(pks),
        }