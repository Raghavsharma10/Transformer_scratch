def from_apps(cls, apps):
        "Takes in an Apps and returns a VersionedProjectState matching it"
        app_models = {}
        for model in apps.get_models(include_swapped=True):
            model_state = VersionedModelState.from_model(model)
            app_models[(model_state.app_label, model_state.name.lower())] = model_state
        return cls(app_models)