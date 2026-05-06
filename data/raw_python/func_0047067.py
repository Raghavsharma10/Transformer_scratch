def _init_map(self):
        """call these all manually because non-cooperative"""
        DecimalAnswerFormRecord._init_map(self)
        DecimalValuesFormRecord._init_map(self)
        TextAnswerFormRecord._init_map(self)
        TextsFormRecord._init_map(self)
        super(edXNumericResponseAnswerFormRecord, self)._init_map()