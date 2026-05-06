def load_all(self):
        """
        Force all reports to be loaded and parsed instead of lazy loading on demand.
        
        :returns: ``self`` or ``None`` if load fails
        """
        try:
            self.toi.load_all()
            self.rosters.load_all()
            #self.summary.load_all()
            self.play_by_play.load_all()
            self.face_off_comp.load_all()
            return self
        except Exception as e:
            print(e)
            return None