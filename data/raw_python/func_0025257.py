def get_api(version: str, ui_version: str=None) -> API_1:
    """Get a versioned interface matching the given version and ui_version.

    version is a string in the form "1.0.2".
    """
    ui_version = ui_version if ui_version else "~1.0"
    return _get_api_with_app(version, ui_version, ApplicationModule.app)