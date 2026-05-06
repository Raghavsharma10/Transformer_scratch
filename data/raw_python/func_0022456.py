def flush(self, save_index=False, save_model=False, clear_buffer=False):
        """Commit all changes, clear all caches."""
        if save_index:
            if self.fresh_index is not None:
                self.fresh_index.save(self.location('index_fresh'))
            if self.opt_index is not None:
                self.opt_index.save(self.location('index_opt'))
        if save_model:
            if self.model is not None:
                self.model.save(self.location('model'))
        self.payload.commit()
        if clear_buffer:
            if hasattr(self, 'fresh_docs'):
                try:
                    self.fresh_docs.terminate() # erase all buffered documents + file on disk
                except:
                    pass
            self.fresh_docs = SqliteDict(journal_mode=JOURNAL_MODE) # buffer defaults to a random location in temp
        self.fresh_docs.sync()