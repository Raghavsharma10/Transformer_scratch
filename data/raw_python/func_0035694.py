def store(self, text, tier):
        """
        Writes text to the underlying Store mapped at tier. If the store doesn't exists, yet, it creates it
        :param text: the text to write
        :param tier: the tier used to identify the store
        :return:
        """
        store = self._stores.get(tier, None)
        if not store:
            store = AutoSplittingFile(self._dir, self._lines_per_store, self._file_name, tier)
            self._stores[tier] = store
        store.write(text)