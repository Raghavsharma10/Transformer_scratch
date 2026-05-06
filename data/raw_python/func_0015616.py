def get_batch_input(self, sentences):
        """Convert sentences to desired input tensors.

        :param sentences: A list of lists representing the input sentences.

        :return word_embd_input, char_embd_input: The desired inputs.
        """
        return get_batch_input(sentences,
                               max_word_len=self.max_word_len,
                               word_dict=self.get_word_dict(),
                               char_dict=self.get_char_dict(),
                               word_ignore_case=self.word_ignore_case,
                               char_ignore_case=self.char_ignore_case)