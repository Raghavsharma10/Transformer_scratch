def as_subbundle(cls, name=None, title=None, title_plural=None):
        """
        Wraps the given bundle so that it can be lazily
        instantiated.

        :param name: The slug for this bundle.
        :param title: The verbose name for this bundle.
        """
        return PromiseBundle(cls, name=name, title=title,
                                title_plural=title_plural)