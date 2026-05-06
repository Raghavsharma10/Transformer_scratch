def _print_split_model(self, path, apps_models):
        """
        Print each model in apps_models into its own file.
        """
        for app, models in apps_models:
            for model in models:
                model_name = model().title
                if self._has_extension(path):
                    model_path = re.sub(r'^(.*)[.](\w+)$', r'\1.%s.%s.\2' % (app, model_name), path)
                else:
                    model_path = '%s.%s.%s.puml' % (path, app, model_name)
                self._print_single_file(model_path, [(app, [model])])