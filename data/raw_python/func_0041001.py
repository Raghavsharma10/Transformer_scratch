def _process_table(self, cache=True):
        """Applies the post processors"""
        table = self.source()
        assert not isinstance(table, None.__class__), \
            "{}.source needs to return something, not None".format(self.__class__.__name__)
        table = post_process(table, self.post_processors())
        if cache:
            self.to_cache(table)
        return table