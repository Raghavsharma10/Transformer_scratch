def get_es(self, default_builder=get_es):
        """Returns the elasticsearch Elasticsearch object to use.

        This uses the django get_es builder by default which takes
        into account settings in ``settings.py``.

        """
        return super(S, self).get_es(default_builder=default_builder)