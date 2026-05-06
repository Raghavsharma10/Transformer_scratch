def tokenize(cls, text, mode='c'):
        """ Converts text into tokens

            :param text: string to be tokenized
            :param mode: split into chars (c) or words (w)
        """
        if mode == 'c':
            return [ch for ch in text]
        else:
            return [w for w in text.split()]