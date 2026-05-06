def isconsistent(self):
        '''Check if the timeseries is consistent'''
        for dt1, dt0 in laggeddates(self):
            if dt1 <= dt0:
                return False
        return True