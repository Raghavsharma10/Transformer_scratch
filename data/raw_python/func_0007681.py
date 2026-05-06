def token_generator(self, texts, **kwargs):
        """Yields tokens from texts as `(text_idx, word)`

        Args:
            texts: The list of texts.
            **kwargs: Supported args include:
                n_threads/num_threads: Number of threads to use. Uses num_cpus - 1 by default.
                batch_size: The number of texts to accumulate into a common working set before processing.
                    (Default value: 1000)
        """
        # Perf optimization. Only process what is necessary.
        n_threads, batch_size = utils._parse_spacy_kwargs(**kwargs)
        nlp = spacy.load(self.lang)

        disabled = ['parser']
        if len(self.exclude_entities) > 0:
            disabled.append('ner')

        kwargs = {
            'batch_size': batch_size,
            'n_threads': n_threads,
            'disable': disabled
        }

        for text_idx, doc in enumerate(nlp.pipe(texts, **kwargs)):
            for word in doc:
                processed_word = self._apply_options(word)
                if processed_word is not None:
                    yield text_idx, processed_word