def _create_markup_model(fixed_language):
    """
    Create a new MarkupItem model that saves itself in a single language.
    """
    title = backend.LANGUAGE_NAMES.get(fixed_language) or fixed_language

    objects = MarkupLanguageManager(fixed_language)

    def save(self, *args, **kwargs):
        self.language = fixed_language
        MarkupItem.save(self, *args, **kwargs)

    class Meta:
        verbose_name = title
        verbose_name_plural = _('%s items') % title
        proxy = True

    classname = "{0}MarkupItem".format(fixed_language.capitalize())

    new_class = type(str(classname), (MarkupItem,), {
        '__module__': MarkupItem.__module__,
        'objects': objects,
        'save': save,
        'Meta': Meta,
    })

    # Make easily browsable
    return new_class