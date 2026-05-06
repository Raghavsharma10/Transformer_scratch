def _loop_wrapper_func(func, args, shared_mem_run, shared_mem_pause, interval, sigint, sigterm, name,
                       logging_level, conn_send, func_running, log_queue):
    """
        to be executed as a separate process (that's why this functions is declared static)
    """
    prefix = get_identifier(name) + ' '

    global log
    log = logging.getLogger(__name__+".log_{}".format(get_identifier(name, bold=False)))
    log.setLevel(logging_level)
    log.addHandler(QueueHandler(log_queue))
  
    sys.stdout = StdoutPipe(conn_send)

    log.debug("enter wrapper_func")

    SIG_handler_Loop(sigint, sigterm, log, prefix)
    func_running.value = True
    
    error = False

    while shared_mem_run.value:
        try:
            # in pause mode, simply sleep
            if shared_mem_pause.value:
                quit_loop = False
            else:
                # if not pause mode -> call func and see what happens
                try:
                    quit_loop = func(*args)
                except LoopInterruptError:
                    raise
                except Exception as e:
                    log.error("error %s occurred in loop calling 'func(*args)'", type(e))
                    log.info("show traceback.print_exc()\n%s", traceback.format_exc())
                    error = True
                    break
 
                if quit_loop is True:
                    log.debug("loop stooped because func returned True")
                    break
 
            time.sleep(interval)
        except LoopInterruptError:
            log.debug("quit wrapper_func due to InterruptedError")
            break

    func_running.value = False
    if error:
        sys.exit(-1)
    else:
        log.debug("wrapper_func terminates gracefully")
    
    # gets rid of the following warnings
    #   Exception ignored in: <_io.FileIO name='/dev/null' mode='rb'>
    #   ResourceWarning: unclosed file <_io.TextIOWrapper name='/dev/null' mode='r' encoding='UTF-8'>
    try:
        if mp.get_start_method() == "spawn":
            sys.stdin.close()
    except AttributeError:
        pass