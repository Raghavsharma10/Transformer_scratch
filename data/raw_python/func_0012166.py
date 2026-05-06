def load(pathtovector,
             wordlist=(),
             num_to_load=None,
             truncate_embeddings=None,
             unk_word=None,
             sep=" "):
        r"""
        Read a file in word2vec .txt format.

        The load function will raise a ValueError when trying to load items
        which do not conform to line lengths.

        Parameters
        ----------
        pathtovector : string
            The path to the vector file.
        header : bool
            Whether the vector file has a header of the type
            (NUMBER OF ITEMS, SIZE OF VECTOR).
        wordlist : iterable, optional, default ()
            A list of words you want loaded from the vector file. If this is
            None (default), all words will be loaded.
        num_to_load : int, optional, default None
            The number of items to load from the file. Because loading can take
            some time, it is sometimes useful to onlyl load the first n items
            from a vector file for quick inspection.
        truncate_embeddings : int, optional, default None
            If this value is not None, the vectors in the vector space will
            be truncated to the number of dimensions indicated by this value.
        unk_word : object
            The object to treat as UNK in your vector space. If this is not
            in your items dictionary after loading, we add it with a zero
            vector.

        Returns
        -------
        r : Reach
            An initialized Reach instance.

        """
        vectors, items = Reach._load(pathtovector,
                                     wordlist,
                                     num_to_load,
                                     truncate_embeddings,
                                     sep)
        if unk_word is not None:
            if unk_word not in set(items):
                unk_vec = np.zeros((1, vectors.shape[1]))
                vectors = np.concatenate([unk_vec, vectors], 0)
                items = [unk_word] + items
                unk_index = 0
            else:
                unk_index = items.index(unk_word)
        else:
            unk_index = None

        return Reach(vectors,
                     items,
                     name=os.path.split(pathtovector)[-1],
                     unk_index=unk_index)