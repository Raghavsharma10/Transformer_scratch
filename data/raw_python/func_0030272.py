def create_cms_plugin_page(apphook, apphook_namespace, placeholder_slot=None):
    """
    Create cms plugin page in all existing languages.
    Add a link to the index page.

    :param apphook: e.g...........: 'FooBarApp'
    :param apphook_namespace: e.g.: 'foobar'
    :return:
    """
    creator = CmsPluginPageCreator(
        apphook=apphook,
        apphook_namespace=apphook_namespace,
    )
    creator.placeholder_slot = placeholder_slot
    plugin_page = creator.create()
    return plugin_page