def sentiment(self):
        """Return a tuple of form (polarity, subjectivity ) where polarity
        is a float within the range [-1.0, 1.0] and subjectivity is a float
        within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is
        very subjective.

        :rtype: named tuple of the form ``Sentiment(polarity=0.0, subjectivity=0.0)``
        """
        #: Enhancement Issue #2
        #: adapted from 'textblob.en.sentiments.py'
        #: Return type declaration
        _RETURN_TYPE = namedtuple('Sentiment', ['polarity', 'subjectivity'])
        _polarity = 0
        _subjectivity = 0
        for s in self.sentences:
            _polarity += s.polarity
            _subjectivity += s.subjectivity
        try:
            polarity = _polarity / len(self.sentences)
        except ZeroDivisionError:
            polarity = 0.0
        try:
            subjectivity = _subjectivity / len(self.sentences)
        except ZeroDivisionError:
            subjectivity = 0.0
        return _RETURN_TYPE(polarity, subjectivity)