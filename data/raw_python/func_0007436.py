def ngram_similarity(samegrams, allgrams, warp=1.0):
        """Similarity for two sets of n-grams.

        :note: ``similarity = (a**e - d**e)/a**e`` where `a` is \
        "all n-grams", `d` is "different n-grams" and `e` is the warp.

        :param samegrams: number of n-grams shared by the two strings.

        :param allgrams: total of the distinct n-grams across the two strings.
        :return: similarity in the range 0.0 to 1.0.

        >>> from ngram import NGram
        >>> NGram.ngram_similarity(5, 10)
        0.5
        >>> NGram.ngram_similarity(5, 10, warp=2)
        0.75
        >>> NGram.ngram_similarity(5, 10, warp=3)
        0.875
        >>> NGram.ngram_similarity(2, 4, warp=2)
        0.75
        >>> NGram.ngram_similarity(3, 4)
        0.75
        """
        if abs(warp - 1.0) < 1e-9:
            similarity = float(samegrams) / allgrams
        else:
            diffgrams = float(allgrams - samegrams)
            similarity = ((allgrams ** warp - diffgrams ** warp)
                    / (allgrams ** warp))
        return similarity