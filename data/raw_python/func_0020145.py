def ExceptionHookPDB(exctype, value, tb):
    '''
    A custom exception handler, with :py:obj:`pdb` post-mortem for debugging.

    '''

    for line in traceback.format_exception_only(exctype, value):
        log.error(line.replace('\n', ''))
    for line in traceback.format_tb(tb):
        log.error(line.replace('\n', ''))
    sys.__excepthook__(exctype, value, tb)
    pdb.pm()