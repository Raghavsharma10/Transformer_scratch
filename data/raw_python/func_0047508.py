def _init_map(self):
        """stub"""
        QuestionTextFormRecord._init_map(self)
        QuestionFilesFormRecord._init_map(self)
        super(QuestionTextAndFilesMixin, self)._init_map()