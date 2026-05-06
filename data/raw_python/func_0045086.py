def _print_app(self, app, models):
        """
        Print the models of app, showing them in a package.
        """
        self._print(self._app_start % app)
        self._print_models(models)
        self._print(self._app_end)