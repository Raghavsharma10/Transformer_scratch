def add_update(self, *args, **kwargs):
        """A records is added, then on subsequent calls, updated"""

        if not self._ai_rec_id:
            self._ai_rec_id = self.add(*args, **kwargs)
        else:
            au_save = self._ai_rec_id
            self.update(*args, **kwargs)
            self._ai_rec_id = au_save

        return self._ai_rec_id