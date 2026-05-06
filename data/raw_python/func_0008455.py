def _create_sentence_objects(self):
        """Returns a list of Sentence objects from the raw text."""
        sentence_objects = []
        sentences = sent_tokenize(self.raw, tokenizer=self.tokenizer)
        char_index = 0  # Keeps track of character index within the blob
        for sent in sentences:
            # Compute the start and end indices of the sentence
            # within the blob. This only works if the sentence splitter
            # does not perform any character replacements or changes to
            # white space.
            # Working: NLTKPunktTokenizer
            # Not working: PatternTokenizer
            try:
                start_index = self.raw.index(sent, char_index)
                char_index += len(sent)
                end_index = start_index + len(sent)
            except ValueError:
                start_index = None
                end_index = None
            # Sentences share the same models as their parent blob
            s = Sentence(
                sent,
                start_index=start_index,
                end_index=end_index,
                tokenizer=self.tokenizer,
                np_extractor=self.np_extractor,
                pos_tagger=self.pos_tagger,
                analyzer=self.analyzer,
                parser=self.parser,
                classifier=self.classifier)
            sentence_objects.append(s)
        return sentence_objects