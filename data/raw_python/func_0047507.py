def _init_metadata(self):
        """stub"""
        ItemTextsFormRecord._init_metadata(self)
        ItemFilesFormRecord._init_metadata(self)
        super(ItemTextsAndFilesMixin, self)._init_metadata()