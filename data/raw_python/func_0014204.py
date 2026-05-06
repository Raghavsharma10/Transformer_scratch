def create_entry_file(self, filename, script_map, enapps):
        '''Creates an entry file for the given script map'''
        if len(script_map) == 0:
            return

        # create the entry file
        template = MakoTemplate('''
<%! import os %>
// dynamic imports are within functions so they don't happen until called
DMP_CONTEXT.loadBundle({
    %for (app, template), script_paths in script_map.items():

    "${ app }/${ template }": () => [
        %for path in script_paths:
        import(/* webpackMode: "eager" */ "./${ os.path.relpath(path, os.path.dirname(filename)) }"),
        %endfor
    ],
    %endfor

})
''')
        content = template.render(
            enapps=enapps,
            script_map=script_map,
            filename=filename,
        ).strip()

        # ensure the parent directories exist
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        # if the file exists, then consider the options
        file_exists = os.path.exists(filename)
        if file_exists and self.running_inline:
            # running inline means that we're in debug mode and webpack is likely watching, so
            # we don't want to recreate the entry file (and cause webpack to constantly reload)
            # unless we have changes
            with open(filename, 'r') as fin:
                if content == fin.read():
                    return False
        if file_exists and not self.options.get('overwrite'):
            raise CommandError('Refusing to destroy existing file: {} (use --overwrite option or remove the file)'.format(filename))

        # if we get here, write the file
        self.message('Creating {}'.format(os.path.relpath(filename, settings.BASE_DIR)), level=3)
        with open(filename, 'w') as fout:
            fout.write(content)
        return True