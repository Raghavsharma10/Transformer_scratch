def _create_markup_plugin(language, model):
    """
    Create a new MarkupPlugin class that represents the plugin type.
    """
    form = type("{0}MarkupItemForm".format(language.capitalize()), (MarkupItemForm,), {
        'default_language': language,
    })

    classname = "{0}MarkupPlugin".format(language.capitalize())
    PluginClass = type(classname, (MarkupPluginBase,), {
        'model': model,
        'form': form,
    })

    return PluginClass