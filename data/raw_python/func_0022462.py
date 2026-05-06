def delete(self, docids):
        """Delete specified documents from the index."""
        logger.info("asked to drop %i documents" % len(docids))
        for index in [self.opt_index, self.fresh_index]:
            if index is not None:
                index.delete(docids)
        self.flush(save_index=True)