def _init_metadata(self):
        """stub"""
        SimpleDifficultyItemFormRecord._init_metadata(self)
        SourceItemFormRecord._init_metadata(self)
        PDFPreviewFormRecord._init_metadata(self)
        PublishedFormRecord._init_metadata(self)
        ProvenanceFormRecord._init_metadata(self)
        super(MecQBankBaseMixin, self)._init_metadata()