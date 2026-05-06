def _init_map(self):
        """Have to call these all separately because they are "end" classes,
        with no super() in them. Non-cooperative."""
        ItemTextsFormRecord._init_map(self)
        ItemFilesFormRecord._init_map(self)
        edXBaseFormRecord._init_map(self)
        IRTItemFormRecord._init_map(self)
        TimeValueFormRecord._init_map(self)
        ProvenanceFormRecord._init_map(self)
        super(edXItemFormRecord, self)._init_map()