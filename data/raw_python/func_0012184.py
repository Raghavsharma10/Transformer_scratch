def _load_fast(filename):
        """Sub for fast loader."""
        it = json.load(open("{}_items.json".format(filename)))
        words, unk_index, name = it["items"], it["unk_index"], it["name"]
        vectors = np.load(open("{}_vectors.npy".format(filename), 'rb'))
        return words, unk_index, name, vectors