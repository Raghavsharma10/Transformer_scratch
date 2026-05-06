def load_fast_format(filename):
        """
        Load a reach instance in fast format.

        As described above, the fast format stores the words and vectors of the
        Reach instance separately, and is drastically faster than loading from
        .txt files.

        Parameters
        ----------
        filename : str
            The filename prefix from which to load. Note that this is not a
            real filepath as such, but a shared prefix for both files.
            In order for this to work, both {filename}_words.json and
            {filename}_vectors.npy should be present.

        """
        words, unk_index, name, vectors = Reach._load_fast(filename)
        return Reach(vectors, words, unk_index=unk_index, name=name)