def add_training_sample(self, text=u'', lang=''):
        """ Initial step for adding new sample to training data.
            You need to call `save_training_samples()` afterwards.

            :param text: Sample text to be added.
            :param lang: Language label for the input text.
        """
        self.trainer.add(text=text, lang=lang)