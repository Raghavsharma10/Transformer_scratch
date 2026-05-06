def save(self, path, write_header=True):
        """
        Save the current vector space in word2vec format.

        Parameters
        ----------
        path : str
            The path to save the vector file to.
        write_header : bool, optional, default True
            Whether to write a word2vec-style header as the first line of the
            file

        """
        with open(path, 'w') as f:

            if write_header:
                f.write(u"{0} {1}\n".format(str(self.vectors.shape[0]),
                        str(self.vectors.shape[1])))

            for i in range(len(self.items)):

                w = self.indices[i]
                vec = self.vectors[i]

                f.write(u"{0} {1}\n".format(w,
                                            " ".join([str(x) for x in vec])))