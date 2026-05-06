def _init_metadata(self):
        """stub"""
        QuestionTextFormRecord._init_metadata(self)
        QuestionFilesFormRecord._init_metadata(self)
        super(QuestionTextAndFilesMixin, self)._init_metadata()