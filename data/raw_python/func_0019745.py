def recmat2egg(recmat, list_length=None):
        """
        Creates egg data object from zero-indexed recall matrix

        Parameters
        ----------
        recmat : list of lists (subs) of lists (encoding lists) of ints or 2D numpy array
            recall matrix representing serial positions of freely recalled words \
            e.g. [[[16, 15, 0, 2, 3, None, None...], [16, 4, 5, 6, 1, None, None...]]]

        list_length : int
            The length of each list (e.g. 16)


        Returns
        ----------
        egg : Egg data object
            egg data object computed from the recall matrix
        """
        from .egg import Egg as Egg

        pres = [[[str(word) for word in list(range(0,list_length))] for reclist in recsub] for recsub in recmat]
        rec = [[[str(word) for word in reclist if word is not None] for reclist in recsub] for recsub in recmat]

        return Egg(pres=pres,rec=rec)