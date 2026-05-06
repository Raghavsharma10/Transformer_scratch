def get_dicts(self):
        """Get word and character dictionaries.

        :return word_dict, char_dict:
        """
        if self.word_dict is None:
            self.word_dict, self.char_dict, self.max_word_len = self.dict_generator(return_dict=True)
        return self.word_dict, self.char_dict