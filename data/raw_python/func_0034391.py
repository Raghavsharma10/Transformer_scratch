def fromRawEntry(cls, **kwargs):
        """
        Helper function to allow wrapping existing data/entries, such as
        those returned by collections.
        """
        id = kwargs["id"]

        kwargs.pop("id")

        what = cls(**kwargs)
        what._new = False
        what.id = id

        return what