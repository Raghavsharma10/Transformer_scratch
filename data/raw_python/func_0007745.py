def _set_seo_models(self, value):
        """ Gets the actual models to be used. """
        seo_models = []
        for model_name in value:
            if "." in model_name:
                app_label, model_name = model_name.split(".", 1)
                model = models.get_model(app_label, model_name)
                if model:
                    seo_models.append(model)
            else:
                app = models.get_app(model_name)
                if app:
                    seo_models.extend(models.get_models(app))
    
        self.seo_models = seo_models