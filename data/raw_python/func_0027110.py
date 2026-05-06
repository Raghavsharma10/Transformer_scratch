def get_extensions(cls):
        """ Get a list of available extensions """
        assemblies = []
        for waldur_extension in pkg_resources.iter_entry_points('waldur_extensions'):
            extension_module = waldur_extension.load()
            if inspect.isclass(extension_module) and issubclass(extension_module, cls):
                if not extension_module.is_assembly():
                    yield extension_module
                else:
                    assemblies.append(extension_module)
        for assembly in assemblies:
            yield assembly