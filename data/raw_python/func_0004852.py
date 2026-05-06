def ydtick(self, dtick, index=1):
        """Set the tick distance."""
        self.layout['yaxis' + str(index)]['dtick'] = dtick
        return self