def file_loc():
    """Return file and line number"""
    import sys
    import inspect
    try:
        raise Exception
    except:
        file_ = '.../' + '/'.join((inspect.currentframe().f_code.co_filename.split('/'))[-3:])
        line_ = sys.exc_info()[2].tb_frame.f_back.f_lineno
        return "{}:{}".format(file_, line_)