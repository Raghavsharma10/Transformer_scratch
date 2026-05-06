def close_stream(self):
        """
        Terminates an open stream and returns the filename
        of the file containing the streamed data.
        """
        if self.fout:
            fout = self.fout
            fout_fn = self.fout_fn
            self.fout.flush()
            self.fout.close()
            self.fout = None
            self.fout_fn = None
            return fout_fn