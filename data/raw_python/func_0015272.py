def needs_fully_loaded(method):
    """Wraps all publicly callable methods of YamlAssistant. If the assistant was loaded
    from cache, this decorator will fully load it first time a publicly callable method
    is used.
    """
    @functools.wraps(method)
    def inner(self, *args, **kwargs):
        if not self.fully_loaded:
            loaded_yaml = yaml_loader.YamlLoader.load_yaml_by_path(self.path)
            self.parsed_yaml = loaded_yaml
            self.fully_loaded = True
        return method(self, *args, **kwargs)

    return inner