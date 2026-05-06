def get_translation_for(package_name: str) -> gettext.NullTranslations:
    """find and return gettext translation for package"""
    localedir = None
    for localedir in pkg_resources.resource_filename(package_name, 'i18n'), None:
        localefile = gettext.find(package_name, localedir)  # type: ignore
        if localefile:
            break
    else:
        pass
    return gettext.translation(package_name, localedir=localedir, fallback=True)