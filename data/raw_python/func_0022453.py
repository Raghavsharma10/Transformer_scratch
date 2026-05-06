def merge(self, other):
        """Merge documents from the other index. Update precomputed similarities
        in the process."""
        other.qindex.normalize, other.qindex.num_best = False, self.topsims
        # update precomputed "most similar" for old documents (in case some of
        # the new docs make it to the top-N for some of the old documents)
        logger.info("updating old precomputed values")
        pos, lenself = 0, len(self.qindex)
        for chunk in self.qindex.iter_chunks():
            for sims in other.qindex[chunk]:
                if pos in self.pos2id:
                    # ignore masked entries (deleted, overwritten documents)
                    docid = self.pos2id[pos]
                    sims = self.sims2scores(sims)
                    self.id2sims[docid] = merge_sims(self.id2sims[docid], sims, self.topsims)
                pos += 1
                if pos % 10000 == 0:
                    logger.info("PROGRESS: updated doc #%i/%i" % (pos, lenself))
        self.id2sims.sync()

        logger.info("merging fresh index into optimized one")
        pos, docids = 0, []
        for chunk in other.qindex.iter_chunks():
            for vec in chunk:
                if pos in other.pos2id: # don't copy deleted documents
                    self.qindex.add_documents([vec])
                    docids.append(other.pos2id[pos])
                pos += 1
        self.qindex.save()
        self.update_ids(docids)

        logger.info("precomputing most similar for the fresh index")
        pos, lenother = 0, len(other.qindex)
        norm, self.qindex.normalize = self.qindex.normalize, False
        topsims, self.qindex.num_best = self.qindex.num_best, self.topsims
        for chunk in other.qindex.iter_chunks():
            for sims in self.qindex[chunk]:
                if pos in other.pos2id:
                    # ignore masked entries (deleted, overwritten documents)
                    docid = other.pos2id[pos]
                    self.id2sims[docid] = self.sims2scores(sims)
                pos += 1
                if pos % 10000 == 0:
                    logger.info("PROGRESS: precomputed doc #%i/%i" % (pos, lenother))
        self.qindex.normalize, self.qindex.num_best = norm, topsims
        self.id2sims.sync()