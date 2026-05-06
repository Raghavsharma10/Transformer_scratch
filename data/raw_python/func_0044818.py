def _show_stat_wrapper_multi_Progress(count, last_count, start_time, max_count, speed_calc_cycles, 
                                      width, q, last_speed, prepend, show_stat_function, len_, 
                                      add_args, lock, info_line, no_move_up=False):
    """
        call the static method show_stat_wrapper for each process
    """
#         print(ESC_BOLD, end='')
#         sys.stdout.flush()
    for i in range(len_):
        _show_stat_wrapper_Progress(count[i], last_count[i], start_time[i], max_count[i], speed_calc_cycles, 
                                    width, q[i], last_speed[i], prepend[i], show_stat_function,
                                    add_args, i, lock[i])
    n = len_
    if info_line is not None:
        s = info_line.value.decode('utf-8')
        s = s.split('\n')
        n += len(s)
        for si in s:
            if width == 'auto':
                width = get_terminal_width()
            if len(si) > width:
                si = si[:width]
            print("{0:<{1}}".format(si, width))
    
    if no_move_up:
        n = 0
                                # this is only a hack to find the end
                                # of the message in a stream
                                # so ESC_HIDDEN+ESC_NO_CHAR_ATTR is a magic ending
    print(terminal.ESC_MOVE_LINE_UP(n) + terminal.ESC_MY_MAGIC_ENDING, end='')
    sys.stdout.flush()