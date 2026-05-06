def run(func, keys, max_procs=None, show_proc=False, affinity=None, **kwargs):
    """
    Provide interface for multiprocessing

    Args:
        func: callable functions
        keys: keys in kwargs that want to use process
        max_procs: max number of processes
        show_proc: whether to show process
        affinity: CPU affinity
        **kwargs: kwargs for func
    """
    if max_procs is None: max_procs = cpu_count()
    kw_arr = saturate_kwargs(keys=keys, **kwargs)
    if len(kw_arr) == 0: return

    if isinstance(affinity, int):
        win32process.SetProcessAffinityMask(win32api.GetCurrentProcess(), affinity)

    task_queue = queue.Queue()
    while len(kw_arr) > 0:
        for _ in range(max_procs):
            if len(kw_arr) == 0: break
            kw = kw_arr.pop(0)
            p = Process(target=func, kwargs=kw)
            p.start()
            sys.stdout.flush()
            task_queue.put(p)
            if show_proc:
                signature = ', '.join([f'{k}={v}' for k, v in kw.items()])
                print(f'[{func.__name__}] ({signature})')
        while not task_queue.empty():
            p = task_queue.get()
            p.join()