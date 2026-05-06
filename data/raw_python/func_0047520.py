def _init_metadata(self):
        """stub"""
        TextAnswerFormRecord._init_metadata(self)
        FilesAnswerFormRecord._init_metadata(self)
        super(AnswerTextAndFilesMixin, self)._init_metadata()