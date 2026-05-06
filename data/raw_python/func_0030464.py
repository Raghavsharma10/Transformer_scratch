def handle_error(self, env):
        '''
        Unhandled exception handler.
        You can put any logging, error warning, etc here.'''
        logger.exception('Exception for %s %s :',
                         env.request.method, env.request.url)