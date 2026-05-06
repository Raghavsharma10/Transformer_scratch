def set_thresh(thresh,p=False,hostname=None):
    '''Sets the level of the threshold slider.
    If ``p==True`` will be interpreted as a _p_-value'''
    driver_send("SET_THRESHNEW %s *%s" % (str(thresh),"p" if p else ""),hostname=hostname)