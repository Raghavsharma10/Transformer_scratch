def fulltext(search, lang=Lang.English, ignore_case=True):
        """Full text search.

        Example::

            filters = Text.fulltext("python pymongo_mate")

        .. note::

            This field doesn't need to specify field.
        """
        return {
            "$text": {
                "$search": search,
                "$language": lang,
                "$caseSensitive": not ignore_case,
                "$diacriticSensitive": False,
            }
        }