def _checkstatus(status, line):
    """Returns state/status after reading the next line.
    The status codes are::
        LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF - BEGIN parsing; 1 - ENTER METADATA GROUP, 2 - READ METADATA LINE,
        3 - END METDADATA GROUP, 4 - END PARSING
    Permitted Transitions::
        LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF --> 1, LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF --> 4
        1 --> 1, 1 --> 2, 1 --> 3
        2 --> 2, 2 --> 3
        3 --> 1, 1 --> 3, 3 --> 4
    """
    newstatus = 0
    if status == 0:
        # begin --> enter metadata group OR end
        if _islinetype(line, GRPSTART):
            newstatus = 1
        elif _isfinal(line):
            newstatus = 4
    elif status == 1:
        # enter metadata group --> enter metadata group
        # OR add metadata item OR leave metadata group
        if _islinetype(line, GRPSTART):
            newstatus = 1
        elif _islinetype(line, GRPEND):
            newstatus = 3
        elif _isassignment(line):
            # test AFTER start and end, as both are also assignments
            newstatus = 2
    elif status == 2:
        if _islinetype(line, GRPEND):
            newstatus = 3
        elif _isassignment(line):
            # test AFTER start and end, as both are also assignments
            newstatus = 2
    elif status == 3:
        if _islinetype(line, GRPSTART):
            newstatus = 1
        elif _islinetype(line, GRPEND):
            newstatus = 3
        elif _isfinal(line):
            newstatus = 4
    if newstatus != 0:
        return newstatus
    elif status != 4:
        raise MTLParseError(
            "Cannot parse the following line after status "
            + "'%s':\n%s" % (STATUSCODE[status], line))