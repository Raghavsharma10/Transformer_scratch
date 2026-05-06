def register_related(cls, related_name):
        """
        Register a related item that should be cloned
        when this model is.

        :param related_name: Use the name you would use in filtering
            i.e.: book not book_set.
        """

        if not hasattr(cls, '_clone_related'):
            cls._clone_related = []

        if type(cls._clone_related) != type([]):
            cls._clone_related = list(cls._clone_related)

        if not related_name in cls._clone_related:
            cls._clone_related.append(related_name)