def _init_map(self):
        """stub"""
        TextAnswerFormRecord._init_map(self)
        FilesAnswerFormRecord._init_map(self)
        super(AnswerTextAndFilesMixin, self)._init_map()