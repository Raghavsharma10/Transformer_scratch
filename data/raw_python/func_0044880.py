def ips_excepthook(excType, excValue, traceback, frame_upcount=0):
    """
    This function is launched after an exception. It launches IPS in suitable frame.
    Also note that if `__mu` is an integer in the local_ns of the closed IPS-Session then another Session
    is launched in the corresponding frame: "__mu" means = "move up" and referes to frame levels.

    :param excType:     Exception type
    :param excValue:    Exception value
    :param traceback:   Traceback
    :param frame_upcount:   int; initial value for diff index; useful if this hook is called from outside
    :return:
    """

    assert isinstance(frame_upcount, int)

    # first: print the traceback:
    tb_printer = TBPrinter(excType, excValue, traceback)

    # go down the stack
    tb = traceback
    tb_frame_list = []
    while tb.tb_next is not None:
        tb_frame_list.append(tb.tb_frame)
        tb = tb.tb_next

    critical_frame = tb.tb_frame
    tb_frame_list.append(critical_frame)

    tb_frame_list.reverse()
    # now the first frame in the list is the critical frame where the exception occured
    index = 0
    diff_index = frame_upcount

    while diff_index is not None:
        index += diff_index
        tb_printer.print(end_offset=index)
        current_frame = tb_frame_list[index]
        diff_index = ip_shell_after_exception(frame=current_frame)