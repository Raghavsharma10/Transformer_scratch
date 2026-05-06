def compare(s1, s2, **kwargs):
        """Compares two strings and returns their similarity.

        :param s1: first string
        :param s2: second string
        :param kwargs: additional keyword arguments passed to __init__.
        :return: similarity between 0.0 and 1.0.

        >>> from ngram import NGram
        >>> NGram.compare('spa', 'spam')
        0.375
        >>> NGram.compare('ham', 'bam')
        0.25
        >>> NGram.compare('spam', 'pam') #N=2
        0.375
        >>> NGram.compare('ham', 'ams', N=1)
        0.5
        """
        if s1 is None or s2 is None:
            if s1 == s2:
                return 1.0
            return 0.0
        try:
            return NGram([s1], **kwargs).search(s2)[0][1]
        except IndexError:
            return 0.0