def classify(self, phrase, cut_to_len=True):
      """ Classify a phrase based on the model. (See corresponding function in PhraseClassifier).
          Provided here mostly to help verify that a created model is worth saving. Technically, the
          results of the training should be enough for that, but it is good to be able to run it on concrete
          examples.
      """
      if (len(phrase) > self.max_phrase_len):
          if not cut_to_len:
              raise Exception("Phrase too long.")
          phrase = phrase[0:self.max_phrase_len]
      if (self.trainer == None):
          raise Exception("Must train the classifier at least once before classifying")
 
      numbers = self.trainer.classify(stringToVector(phrase, self.vocab, self.max_vector_len))
      return zip(self.targetTranslate, numbers)