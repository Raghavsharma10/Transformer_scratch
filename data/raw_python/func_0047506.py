def _init_map(self):
        """stub"""
        ItemTextsFormRecord._init_map(self)
        ItemFilesFormRecord._init_map(self)
        super(ItemTextsAndFilesMixin, self)._init_map()