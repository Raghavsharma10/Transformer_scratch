def from_keyword_list(self, keyword_list, strictness=2, timeout=3):
        """
        Convert a list of keywords to sentence. The result is sometimes None

        :param list keyword_list: a list of string
        :param int | None strictness: None for highest strictness. 2 or 1 for a less strict POS matching
        :param float timeout: timeout of this function
        :return list of tuple: sentence generated

        >>> SentenceMaker().from_keyword_list(['Love', 'blind', 'trouble'])
        [('For', False), ('love', True), ('to', False), ('such', False), ('blind', True), ('we', False), ('must', False), ('turn', False), ('to', False), ('the', False), ('trouble', True)]
        """
        keyword_tags = nltk.pos_tag(keyword_list)

        start = time()
        while time() - start < timeout:
            index = 0
            output_list = []
            tagged_sent = self.random_sentences.get_tagged_sent()
            for word, tag in tagged_sent:
                if index >= len(keyword_tags):
                    return self.get_overlap(keyword_list, output_list, is_word_list=True)

                if self.match_pos(tag, keyword_tags[index][1], strictness=strictness):
                    output_list.append(keyword_tags[index][0])
                    index += 1
                else:
                    output_list.append(word)

        return []