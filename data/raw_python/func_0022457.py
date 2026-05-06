def close(self):
        """Explicitly close open file handles, databases etc."""
        try:
            self.payload.close()
        except:
            pass
        try:
            self.model.close()
        except:
            pass
        try:
            self.fresh_index.close()
        except:
            pass
        try:
            self.opt_index.close()
        except:
            pass
        try:
            self.fresh_docs.terminate()
        except:
            pass