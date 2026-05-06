def _init_metadata(self):
        """Have to call these all separately because they are "end" classes,
        with no super() in them. Non-cooperative."""
        ItemTextsFormRecord._init_metadata(self)
        ItemFilesFormRecord._init_metadata(self)
        edXBaseFormRecord._init_metadata(self)
        IRTItemFormRecord._init_metadata(self)
        TimeValueFormRecord._init_metadata(self)
        ProvenanceFormRecord._init_metadata(self)
        super(edXItemFormRecord, self)._init_metadata()