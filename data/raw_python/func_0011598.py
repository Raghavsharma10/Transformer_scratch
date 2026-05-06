def field_pk_to_json(self, model, pk):
        """Convert a primary key to a JSON dict."""
        app_label = model._meta.app_label
        model_name = model._meta.model_name
        return {
            'app': app_label,
            'model': model_name,
            'pk': pk,
        }