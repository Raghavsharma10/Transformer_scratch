def add(self, sentence_text, **kwargs):
        ''' Parse a text string and add it to this doc '''
        sent = MeCabSent.parse(sentence_text, **kwargs)
        self.sents.append(sent)
        return sent