def compile_mako_files(self, app_config):
        '''Compiles the Mako templates within the apps of this system'''
        # go through the files in the templates, scripts, and styles directories
        for subdir_name in self.SEARCH_DIRS:
            subdir = subdir_name.format(
                app_path=app_config.path,
                app_name=app_config.name,
            )

            def recurse_path(path):
                self.message('searching for Mako templates in {}'.format(path), 1)
                if os.path.exists(path):
                    for filename in os.listdir(path):
                        filepath = os.path.join(path, filename)
                        _, ext = os.path.splitext(filename)
                        if filename.startswith('__'):  # __dmpcache__, __pycache__
                            continue

                        elif os.path.isdir(filepath):
                            recurse_path(filepath)

                        elif ext.lower() in ( '.htm', '.html', '.mako' ):
                            # create the template object, which creates the compiled .py file
                            self.message('compiling {}'.format(filepath), 2)
                            try:
                                get_template_for_path(filepath)
                            except TemplateSyntaxError:
                                if not self.options.get('ignore_template_errors'):
                                    raise

            recurse_path(subdir)