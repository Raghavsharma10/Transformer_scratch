def ExceptionHook(exctype, value, tb):
    '''
    A custom exception handler that logs errors to file.

    '''

    for line in traceback.format_exception_only(exctype, value):
        log.error(line.replace('\n', ''))
    for line in traceback.format_tb(tb):
        log.error(line.replace('\n', ''))
    sys.__excepthook__(exctype, value, tb)