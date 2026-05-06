def _process(self, input):
        '''
        Takes in html-mixed body text as a string and returns a list of strings,
        lower case and with punctuation given spacing. 

        Called by self._gen_sentence()

        Args:
            inpnut (string): body text
        '''

        input = re.sub("<[^>]*>", " ", input) 
        punct = list(string.punctuation)
        for symbol in punct:
            input = input.replace(symbol, " %s " % symbol)
        input = filter(lambda x: x != u'', input.lower().split(' '))
        return input