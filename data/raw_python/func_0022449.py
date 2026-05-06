def sims2scores(self, sims, eps=1e-7):
        """Convert raw similarity vector to a list of (docid, similarity) results."""
        result = []
        if isinstance(sims, numpy.ndarray):
            sims = abs(sims) # TODO or maybe clip? are opposite vectors "similar" or "dissimilar"?!
            for pos in numpy.argsort(sims)[::-1]:
                if pos in self.pos2id and sims[pos] > eps: # ignore deleted/rewritten documents
                    # convert positions of resulting docs back to ids
                    result.append((self.pos2id[pos], sims[pos]))
                    if len(result) == self.topsims:
                        break
        else:
            for pos, score in sims:
                if pos in self.pos2id and abs(score) > eps: # ignore deleted/rewritten documents
                    # convert positions of resulting docs back to ids
                    result.append((self.pos2id[pos], abs(score)))
                    if len(result) == self.topsims:
                        break
        return result