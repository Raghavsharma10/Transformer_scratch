def _process_worker(call_queue, result_queue, shutdown):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a seperate process.

    Args:
        call_queue: A multiprocessing.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A multiprocessing.Queue of _ResultItems that will written
            to by the worker.
        shutdown: A multiprocessing.Event that will be set as a signal to the
            worker that it should exit when call_queue is empty.
    """
    while True:
        try:
            call_item = call_queue.get(block=True, timeout=0.1)
        except queue.Empty:
            if shutdown.is_set():
                return
        else:
            try:
                r = call_item()
            except BaseException as e:
                result_queue.put(_ResultItem(call_item.work_id, exception=e))
            else:
                result_queue.put(_ResultItem(call_item.work_id, result=r))