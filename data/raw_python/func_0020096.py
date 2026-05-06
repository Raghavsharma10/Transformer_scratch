def exception_handler(self, pdb):
        '''
        A custom exception handler.

        :param pdb: If :py:obj:`True`, enters PDB post-mortem \
               mode for debugging.

        '''

        # Grab the exception
        exctype, value, tb = sys.exc_info()

        # Log the error and create a .err file
        errfile = os.path.join(self.dir, self.name + '.err')
        with open(errfile, 'w') as f:
            for line in traceback.format_exception_only(exctype, value):
                ln = line.replace('\n', '')
                log.error(ln)
                print(ln, file=f)
            for line in traceback.format_tb(tb):
                ln = line.replace('\n', '')
                log.error(ln)
                print(ln, file=f)

        # Re-raise?
        if pdb:
            raise