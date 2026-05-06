def is_mention_line(cls, word):
        """ Detects links and mentions

            :param word: Token to be evaluated
        """
        if word.startswith('@'):
            return True
        elif word.startswith('http://'):
            return True
        elif word.startswith('https://'):
            return True
        else:
            return False