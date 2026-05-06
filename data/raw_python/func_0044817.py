def _show_stat_wrapper_Progress(count, last_count, start_time, max_count, speed_calc_cycles, 
                                width, q, last_speed, prepend, show_stat_function, add_args,
                                i, lock):
    """
        calculate 
    """
    count_value, max_count_value, speed, tet, ttg, = Progress._calc(count, 
                                                                    last_count, 
                                                                    start_time, 
                                                                    max_count, 
                                                                    speed_calc_cycles, 
                                                                    q,
                                                                    last_speed, 
                                                                    lock) 
    return show_stat_function(count_value, max_count_value, prepend, speed, tet, ttg, width, i, **add_args)