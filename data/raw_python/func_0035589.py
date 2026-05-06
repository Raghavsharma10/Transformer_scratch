def breakRankTies(self, oldsym, newsym):
        """break Ties to form a new list with the same integer ordering
        from high to low

        Example
        old = [ 4, 2, 4, 7, 8]  (Two ties, 4 and 4)
        new = [60, 2 61,90,99]
        res = [ 4, 0, 3, 1, 2]
                *     *        This tie is broken in this case
        """
        stableSort = map(None, oldsym, newsym, range(len(oldsym)))
        stableSort.sort()

        lastOld, lastNew = None, None
        x = -1
        for old, new, index in stableSort:
            if old != lastOld:
                x += 1
                # the last old value was changed, so update both
                lastOld = old
                lastNew = new
            elif new != lastNew:
                # break the tie based on the new info (update lastNew)
                x += 1
                lastNew = new
            newsym[index] = x