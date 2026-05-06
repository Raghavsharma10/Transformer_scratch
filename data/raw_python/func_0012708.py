def tokenize_list(self, text):
        """
        Split a text into separate words.
        """
        return [self.get_record_token(record) for record in self.analyze(text)]