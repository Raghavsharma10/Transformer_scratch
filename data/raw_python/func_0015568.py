def pyError(self, type, value, traceback):
        '''Handles an error thrown during invocation of an EWrapper method.

        Arguments are those provided by sys.exc_info()
        '''
        sys.stderr.write("Exception thrown during EWrapper method dispatch:\n")
        print_exception(type, value, traceback)