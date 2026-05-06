def _init_map(self):
        """stub"""
        SimpleDifficultyItemFormRecord._init_map(self)
        SourceItemFormRecord._init_map(self)
        PDFPreviewFormRecord._init_map(self)
        PublishedFormRecord._init_map(self)
        ProvenanceFormRecord._init_map(self)
        super(MecQBankBaseMixin, self)._init_map()