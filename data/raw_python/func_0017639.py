def model_page(self, request, app_label, model_name, rest_of_url=None):
        """
        Handles the model-specific functionality of the databrowse site,
        delegating<to the appropriate ModelDatabrowse class.
        """
        try:
            model = get_model(app_label, model_name)
        except LookupError:
            model = None

        if model is None:
            raise http.Http404("App %r, model %r, not found." %
                               (app_label, model_name))
        try:
            databrowse_class = self.registry[model]
        except KeyError:
            raise http.Http404("This model exists but has not been registered "
                               "with databrowse.")
        return databrowse_class(model, self).root(request, rest_of_url)