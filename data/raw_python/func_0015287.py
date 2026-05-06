def _ass_refresh_attrs(self, cached_ass, file_ass):
        """Completely refreshes cached assistant from file.

        Args:
            cached_ass: an assistant from cache hierarchy
                        (for format see Cache class docstring)
            file_ass: the respective assistant from filesystem hierarchy
                      (for format see what refresh_role accepts)
        """
        # we need to process assistant in custom way to see unexpanded args, etc.
        loaded_ass = yaml_loader.YamlLoader.load_yaml_by_path(file_ass['source'], log_debug=True)
        attrs = loaded_ass
        yaml_checker.check(file_ass['source'], attrs)
        cached_ass['source'] = file_ass['source']
        cached_ass['ctime'] = os.path.getctime(file_ass['source'])
        cached_ass['attrs'] = {}
        cached_ass['snippets'] = {}
        # only cache these attributes if they're actually found in assistant
        # we do this to specify the default values for them just in one place
        # which is currently YamlAssistant.parsed_yaml property setter
        for a in ['fullname', 'description', 'icon_path']:
            if a in attrs:
                cached_ass['attrs'][a] = attrs.get(a)
        # args have different processing, we can't just take them from assistant
        if 'args' in attrs:
            cached_ass['attrs']['args'] = {}
        for argname, argparams in attrs.get('args', {}).items():
            if 'use' in argparams or 'snippet' in argparams:
                snippet_name = argparams.pop('use', None) or argparams.pop('snippet')
                snippet = yaml_snippet_loader.YamlSnippetLoader.get_snippet_by_name(snippet_name)
                cached_ass['attrs']['args'][argname] = snippet.get_arg_by_name(argname)
                cached_ass['attrs']['args'][argname].update(argparams)
                cached_ass['snippets'][snippet.name] = self._get_snippet_ctime(snippet.name)
            else:
                cached_ass['attrs']['args'][argname] = argparams