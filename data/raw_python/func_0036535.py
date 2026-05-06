def simhash(self, content):
        """
        Select policies for simhash on the different types of content.
        """
        if content is None:
            self.hash = -1
            return

        if isinstance(content, str):
            features = self.tokenizer_func(content, self.keyword_weight_pari)
            self.hash = self.build_from_features(features)
        elif isinstance(content, collections.Iterable):
            self.hash = self.build_from_features(content)
        elif isinstance(content, int):
            self.hash = content
        else:
            raise Exception("Unsupported parameter type %s" % type(content))