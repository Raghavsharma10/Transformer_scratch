def model_from_list(l, header):
        """Return a model with a collection from a list of entry"""
        col = groups.sortableListe(PseudoAccesCategorie(n) for n in l)
        return MultiSelectModel(col, header)