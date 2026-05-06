def on_builder_inited(app):
    """
    Hooks into Sphinx's ``builder-inited`` event.

    Builds out the ReST API source.
    """
    config = app.builder.config

    target_directory = (
        pathlib.Path(app.builder.env.srcdir) / config.uqbar_api_directory_name
    )

    initial_source_paths: List[str] = []
    source_paths = config.uqbar_api_source_paths
    for source_path in source_paths:
        if isinstance(source_path, types.ModuleType):
            if hasattr(source_path, "__path__"):
                initial_source_paths.extend(getattr(source_path, "__path__"))
            else:
                initial_source_paths.extend(source_path.__file__)
            continue
        try:
            module = importlib.import_module(source_path)
            if hasattr(module, "__path__"):
                initial_source_paths.extend(getattr(module, "__path__"))
            else:
                initial_source_paths.append(module.__file__)
        except ImportError:
            initial_source_paths.append(source_path)

    root_documenter_class = config.uqbar_api_root_documenter_class
    if isinstance(root_documenter_class, str):
        module_name, _, class_name = root_documenter_class.rpartition(".")
        module = importlib.import_module(module_name)
        root_documenter_class = getattr(module, class_name)

    module_documenter_class = config.uqbar_api_module_documenter_class
    if isinstance(module_documenter_class, str):
        module_name, _, class_name = module_documenter_class.rpartition(".")
        module = importlib.import_module(module_name)
        module_documenter_class = getattr(module, class_name)

    # Don't modify the list in Sphinx's config. Sphinx won't pickle class
    # references, and strips them from the saved config. That leads to Sphinx
    # believing that the config has changed on every run.
    member_documenter_classes = list(config.uqbar_api_member_documenter_classes or [])
    for i, member_documenter_class in enumerate(member_documenter_classes):
        if isinstance(member_documenter_class, str):
            module_name, _, class_name = member_documenter_class.rpartition(".")
            module = importlib.import_module(module_name)
            member_documenter_classes[i] = getattr(module, class_name)

    api_builder = uqbar.apis.APIBuilder(
        initial_source_paths=initial_source_paths,
        target_directory=target_directory,
        document_empty_modules=config.uqbar_api_document_empty_modules,
        document_private_members=config.uqbar_api_document_private_members,
        document_private_modules=config.uqbar_api_document_private_modules,
        member_documenter_classes=member_documenter_classes or None,
        module_documenter_class=module_documenter_class,
        root_documenter_class=root_documenter_class,
        title=config.uqbar_api_title,
        logger_func=logger_func,
    )
    api_builder()