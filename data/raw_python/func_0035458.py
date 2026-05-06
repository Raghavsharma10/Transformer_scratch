def is_timeout(self):
        '''
        Check if the lapse between initialization and now is more than ``self.timeout``.
        '''
        lapse = datetime.datetime.now() - self.init_time
        return lapse > datetime.timedelta(seconds=self.timeout)