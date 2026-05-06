def get_es(self):
        """Returns an `Elasticsearch`.

        * If there's an s, then it returns that `Elasticsearch`.
        * If the es was provided in the constructor, then it returns
          that `Elasticsearch`.
        * Otherwise, it creates a new `Elasticsearch` and returns
          that.

        Override this if that behavior isn't correct for you.

        """
        if self.s:
            return self.s.get_es()

        return self.es or get_es()