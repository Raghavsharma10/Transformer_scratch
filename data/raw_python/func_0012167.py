def _load(pathtovector,
              wordlist,
              num_to_load=None,
              truncate_embeddings=None,
              sep=" "):
        """Load a matrix and wordlist from a .vec file."""
        vectors = []
        addedwords = set()
        words = []

        try:
            wordlist = set(wordlist)
        except ValueError:
            wordlist = set()

        logger.info("Loading {0}".format(pathtovector))

        firstline = open(pathtovector).readline().strip()
        try:
            num, size = firstline.split(sep)
            num, size = int(num), int(size)
            logger.info("Vector space: {} by {}".format(num, size))
            header = True
        except ValueError:
            size = len(firstline.split(sep)) - 1
            logger.info("Vector space: {} dim, # items unknown".format(size))
            word, rest = firstline.split(sep, 1)
            # If the first line is correctly parseable, set header to False.
            header = False

        if truncate_embeddings is None or truncate_embeddings == 0:
            truncate_embeddings = size

        for idx, line in enumerate(open(pathtovector, encoding='utf-8')):

            if header and idx == 0:
                continue

            word, rest = line.rstrip(" \n").split(sep, 1)

            if wordlist and word not in wordlist:
                continue

            if word in addedwords:
                raise ValueError("Duplicate: {} on line {} was in the "
                                 "vector space twice".format(word, idx))

            if len(rest.split(sep)) != size:
                raise ValueError("Incorrect input at index {}, size "
                                 "is {}, expected "
                                 "{}".format(idx+1,
                                             len(rest.split(sep)), size))

            words.append(word)
            addedwords.add(word)
            vectors.append(np.fromstring(rest, sep=sep)[:truncate_embeddings])

            if num_to_load is not None and len(addedwords) >= num_to_load:
                break

        vectors = np.array(vectors).astype(np.float32)

        logger.info("Loading finished")
        if wordlist:
            diff = wordlist - addedwords
            if diff:
                logger.info("Not all items from your wordlist were in your "
                            "vector space: {}.".format(diff))

        return vectors, words