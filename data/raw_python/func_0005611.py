def print_progress_bar_multi_threads(nb_threads, suffix='', decimals=1, length=15,
                                     fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    string = ""
    for k in range(nb_threads):
        try:
            threads_state = eval(read_file("threads_state_%s" % str(k)))
        except SyntaxError:
            time.sleep(0.001)
            try:
                threads_state = eval(read_file("threads_state_%s" % str(k)))
            except SyntaxError:
                pass

        iteration = threads_state["iteration"]
        total = threads_state["total"]
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        # filled_length = int(length * iteration // total)
        # bar = fill * filled_length + '-' * (length - filled_length)
        prefix = "Thread %s :" % str(k)
        string = string + '%s %s%% ' % (prefix, percent)

    print(string + " " + suffix)