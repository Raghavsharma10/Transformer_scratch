def generate_script_map(self, config):
        '''
        Maps templates in this app to their scripts.  This function deep searches
        app/templates/* for the templates of this app.  Returns the following
        dictionary with absolute paths:

        {
            ( 'appname', 'template1' ): [ '/abs/path/to/scripts/template1.js', '/abs/path/to/scripts/supertemplate1.js' ],
            ( 'appname', 'template2' ): [ '/abs/path/to/scripts/template2.js', '/abs/path/to/scripts/supertemplate2.js', '/abs/path/to/scripts/supersuper2.js' ],
            ...
        }

        Any files or subdirectories starting with double-underscores (e.g. __dmpcache__) are skipped.
        '''
        script_map = OrderedDict()
        template_root = os.path.join(os.path.relpath(config.path, settings.BASE_DIR), 'templates')
        def recurse(folder):
            subdirs = []
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    if filename.startswith('__'):
                        continue
                    filerel = os.path.join(folder, filename)
                    if os.path.isdir(filerel):
                        subdirs.append(filerel)

                    elif os.path.isfile(filerel):
                        template_name = os.path.relpath(filerel, template_root)
                        scripts = self.template_scripts(config, template_name)
                        key = ( config.name, os.path.splitext(template_name)[0] )
                        self.message('Found template: {}; static files: {}'.format(key, scripts), level=3)
                        script_map[key] = scripts

            for subdir in subdirs:
                recurse(subdir)

        recurse(template_root)
        return script_map