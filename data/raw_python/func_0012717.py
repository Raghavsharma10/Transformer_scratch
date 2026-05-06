def get_record_pos(self, record):
        """
        Given a record, get the word's part of speech.

        Here we're going to return MeCab's part of speech (written in
        Japanese), though if it's a stopword we prefix the part of speech
        with '~'.
        """
        if self.is_stopword_record(record):
            return '~' + record.pos
        else:
            return record.pos