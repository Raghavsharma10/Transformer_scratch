def add_to_dict(self, text):
        """ Generate word n-tuple and next word probability dict """
        n = self.n

        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
        # '' is a special symbol for the start of a sentence like pymarkovchain uses
        for sentence in sentences:
            sentence = sentence.replace('"','') # remove quotes
            words = sentence.strip().split()  # split each sentence into its constituent words
            if len(words) == 0:
                continue

            # first word follows a sentence end
            self.word_dict[("",)][words[0]].count += 1

            for j in range(1, n+1):
                for i in range(len(words) - 1):
                    if i + j >= len(words):
                        continue
                    word = tuple(words[i:i + j])
                    self.word_dict[word][words[i + j]].count += 1

                # last word precedes a sentence end
                self.word_dict[tuple(words[len(words) - j:len(words)])][""].count += 1

        # We've now got the db filled with parametrized word counts
        # We still need to normalize this to represent probabilities
        for word in self.word_dict:
            wordsum = 0
            for nextword in self.word_dict[word]:
                wordsum += self.word_dict[word][nextword].count
            if wordsum != 0:
                for nextword in self.word_dict[word]:
                    self.word_dict[word][nextword].prob = self.word_dict[word][nextword].count / wordsum