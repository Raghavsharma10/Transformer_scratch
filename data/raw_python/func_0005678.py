def get_translation_functions(package_name: str, names: Tuple[str, ...] = ('gettext',)):
    """finds and installs translation functions for package"""
    translation = get_translation_for(package_name)
    return [getattr(translation, x) for x in names]