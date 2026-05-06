def classify(self, phrase, cut_to_len=True):
      """ Classify a phrase based on the loaded model. If cut_to_len is True, cut to
          desired length."""
      if (len(phrase) > self.max_phrase_len):
          if not cut_to_len:
              raise Exception("Phrase too long.")
          phrase = phrase[0:self.max_phrase_len]

      numbers = self.classifier.classify(stringToVector(phrase, self.vocab, self.max_vector_len))
      return zip(self.targets, numbers)