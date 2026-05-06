def read_stats(self):
        """ Reads the statistics view from IXN and saves it in statistics dictionary. """

        captions, rows = self._get_pages()
        name_caption_index = captions.index(self.name_caption)
        captions.pop(name_caption_index)
        self.captions = captions
        self.statistics = OrderedDict()
        for row in rows:
            name = row.pop(name_caption_index)
            self.statistics[name] = row