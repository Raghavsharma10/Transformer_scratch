def _init_map(self):
        """stub"""
        QuestionTextsFormRecord._init_map(self)
        QuestionFilesFormRecord._init_map(self)
        super(QuestionTextsAndFilesMixin, self)._init_map()