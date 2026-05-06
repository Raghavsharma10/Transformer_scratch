def extract_features(self, text):
        """Extracts features from a body of text.

        :rtype: dictionary of features

        """
        # Feature extractor may take one or two arguments
        try:
            return self.feature_extractor(text, self.train_set)
        except (TypeError, AttributeError):
            return self.feature_extractor(text)