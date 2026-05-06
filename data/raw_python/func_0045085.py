def _print_single_file(self, path, apps_models):
        """
        Print apps_models which contains a list of 2-tuples containing apps and their models
        into a single file.
        """
        if path:
            outfile = codecs.open(path, 'w', encoding='utf-8')
            self._print = lambda s: outfile.write(s + '\n')
        self._print(self._diagram_start)
        for app, app_models in apps_models:
            self._print_app(app, app_models)
        self._print(self._diagram_end)
        if path:
            outfile.close()