def save_fast_format(self, filename):
        """
        Save a reach instance in a fast format.

        The reach fast format stores the words and vectors of a Reach instance
        separately in a JSON and numpy format, respectively.

        Parameters
        ----------
        filename : str
            The prefix to add to the saved filename. Note that this is not the
            real filename under which these items are stored.
            The words and unk_index are stored under "{filename}_words.json",
            and the numpy matrix is saved under "{filename}_vectors.npy".

        """
        items, _ = zip(*sorted(self.items.items(), key=lambda x: x[1]))
        items = {"items": items,
                 "unk_index": self.unk_index,
                 "name": self.name}

        json.dump(items, open("{}_items.json".format(filename), 'w'))
        np.save(open("{}_vectors.npy".format(filename), 'wb'), self.vectors)