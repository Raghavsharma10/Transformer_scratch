def close(self):
        '''Releasing hardware resources.
        '''
        try:
            self.dut.close()
        except Exception:
            logging.warning('Closing DUT was not successful')
        else:
            logging.debug('Closed DUT')