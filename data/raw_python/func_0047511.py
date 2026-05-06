def _init_metadata(self):
        """stub"""
        QuestionTextsFormRecord._init_metadata(self)
        QuestionFilesFormRecord._init_metadata(self)
        super(QuestionTextsAndFilesMixin, self)._init_metadata()