def field_pk_from_json(self, data):
        """Load a PkOnlyModel from a JSON dict."""
        model = get_model(data['app'], data['model'])
        return PkOnlyModel(self, model, data['pk'])