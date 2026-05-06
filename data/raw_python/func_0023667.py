def run(self):
        '''Run the callback periodically'''
        while not self.wait(self.delay()):
            try:
                logger.info('Invoking callback %s', self.callback)
                self.callback()
            except StandardError:
                logger.exception('Callback failed')