def getMapScale(self, latitude, level, dpi=96):
        '''
        returns the map scale on the dpi of the screen
        '''
        dpm = dpi / 0.0254  # convert to dots per meter
        return self.getGroundResolution(latitude, level) * dpm