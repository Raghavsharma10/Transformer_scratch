def _init_map(self):
        """stub"""
        QuestionTextAndFilesMixin._init_map(self)
        BaseMultiChoiceTextQuestionFormRecord._init_map(self)
        super(MultiChoiceTextAndFilesQuestionFormRecord, self)._init_map()