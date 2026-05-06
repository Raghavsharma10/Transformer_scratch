def setup(app) -> Dict[str, Any]:
    """
    Sets up Sphinx extension.
    """
    app.add_config_value("uqbar_api_directory_name", "api", "env")
    app.add_config_value("uqbar_api_document_empty_modules", False, "env")
    app.add_config_value("uqbar_api_document_private_members", False, "env")
    app.add_config_value("uqbar_api_document_private_modules", False, "env")
    app.add_config_value("uqbar_api_member_documenter_classes", None, "env")
    app.add_config_value("uqbar_api_module_documenter_class", None, "env")
    app.add_config_value("uqbar_api_root_documenter_class", None, "env")
    app.add_config_value("uqbar_api_source_paths", None, "env")
    app.add_config_value("uqbar_api_title", "API", "html")
    app.connect("builder-inited", on_builder_inited)
    return {
        "version": uqbar.__version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }