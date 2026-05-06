def _init_metadata(self):
        """stub"""
        MultiChoiceAnswerFormRecord._init_metadata(self)
        FilesAnswerFormRecord._init_metadata(self)
        FeedbackAnswerFormRecord._init_metadata(self)
        super(MultiChoiceFeedbackAndFilesAnswerFormRecord, self)._init_metadata()