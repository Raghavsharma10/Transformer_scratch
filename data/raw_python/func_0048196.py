def _init_map(self):
        """stub"""
        super(edXNumericResponseQuestionFormRecord, self)._init_map()
        QuestionTextFormRecord._init_map(self)
        QuestionFilesFormRecord._init_map(self)
        self.my_osid_object_form._my_map['text']['text'] = ''