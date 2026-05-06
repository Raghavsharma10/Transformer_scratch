def _print_split_app(self, path, apps_models):
        """
        Print each app in apps_models associative list into its own file.
        """
        for app, models in apps_models:
            # Convert dir/file.puml to dir/file.app.puml to print to an app specific file
            if self._has_extension(path):
                app_path = re.sub(r'^(.*)[.](\w+)$', r'\1.%s.\2' % app, path)
            else:
                app_path = '%s.%s.puml' % (path, app)

            self._print_single_file(app_path, [(app, models)])