def update_dicts(self, sentence):
        """Add new sentence to generate dictionaries.

        :param sentence: A list of strings representing the sentence.
        """
        self.dict_generator(sentence=sentence)
        self.word_dict, self.char_dict = None, None