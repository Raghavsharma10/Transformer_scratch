def load_datafile(self, name, search_path=None, **kwargs):
        """
        find datafile and load them from codec
        """
        if not search_path:
            search_path = self.define_dir

        self.debug_msg('loading datafile %s from %s' % (name, str(search_path)))
        return codec.load_datafile(name, search_path, **kwargs)