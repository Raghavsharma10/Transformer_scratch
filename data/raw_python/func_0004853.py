def ynticks(self, nticks, index=1):
        """Set the number of ticks."""
        self.layout['yaxis' + str(index)]['nticks'] = nticks
        return self