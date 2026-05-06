def connection_checker(self):
        '''Run periodic reconnection checks'''
        thread = ConnectionChecker(self)
        logger.info('Starting connection-checker thread')
        thread.start()
        try:
            yield thread
        finally:
            logger.info('Stopping connection-checker')
            thread.stop()
            logger.info('Joining connection-checker')
            thread.join()