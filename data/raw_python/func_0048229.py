def _init_metadata(self):
        """stub"""
        QuestionTextAndFilesMixin._init_metadata(self)
        BaseMultiChoiceTextQuestionFormRecord._init_metadata(self)
        super(MultiChoiceTextAndFilesQuestionFormRecord, self)._init_metadata()