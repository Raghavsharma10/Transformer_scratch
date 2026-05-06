def classify(self, text):
        """Classifies the text.

        :param str text: A string of text.

        """
        text_features = self.extract_features(text)
        return self.classifier.classify(text_features)