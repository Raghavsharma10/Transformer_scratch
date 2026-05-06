def models(self):
        """Return self.application models."""
        Model_ = self.app.config['PEEWEE_MODELS_CLASS']
        ignore = self.app.config['PEEWEE_MODELS_IGNORE']

        models = []
        if Model_ is not Model:
            try:
                mod = import_module(self.app.config['PEEWEE_MODELS_MODULE'])
                for model in dir(mod):
                    models = getattr(mod, model)
                    if not isinstance(model, pw.Model):
                        continue
                    models.append(models)
            except ImportError:
                return models
        elif isinstance(Model_, BaseSignalModel):
            models = BaseSignalModel.models

        return [m for m in models if m._meta.name not in ignore]