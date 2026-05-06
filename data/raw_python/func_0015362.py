def get_snippet_by_name(cls, name):
        """name is in dotted format, e.g. topsnippet.something.wantedsnippet"""
        name_with_dir_separators = name.replace('.', os.path.sep)
        loaded = yaml_loader.YamlLoader.load_yaml_by_relpath(cls.snippets_dirs,
                                                             name_with_dir_separators + '.yaml')
        if loaded:
            return cls._create_snippet(name, *loaded)

        raise exceptions.SnippetNotFoundException('no such snippet: {name}'.
                                                  format(name=name_with_dir_separators))