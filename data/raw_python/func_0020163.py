def dvs(self):
        '''
        Shows the data validation summary (DVS) for the target.

        '''

        DVS(self.ID, season=self.season, mission=self.mission,
            model=self.model_name, clobber=self.clobber)