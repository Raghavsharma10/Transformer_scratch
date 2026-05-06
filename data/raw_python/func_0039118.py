def stop(self):
        """Stop the progress bar."""
        if self._progressing:
            self._progressing = False
            self._thread.join()