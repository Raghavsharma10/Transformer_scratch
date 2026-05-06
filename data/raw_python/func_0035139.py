def _frames(traceback):
    '''
    Returns generator that iterates over frames in a traceback
    '''
    frame = traceback
    while frame.tb_next:
      frame = frame.tb_next
      yield frame.tb_frame
    return