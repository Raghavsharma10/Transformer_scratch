def _push(host, port, q, done, mps, stop, test_mode):
    """Worker thread. Connect to host/port, pull data from q until done is set"""
    sock = None
    retry_line = None
    while not ( stop.is_set() or ( done.is_set() and retry_line == None and q.empty()) ):
        stime = time.time()

        if sock == None and not test_mode:
            sock = _mksocket(host, port, q, done, stop)
            if sock == None:
                break

        if retry_line:
            line = retry_line
            retry_line = None
        else:
            try:
                line = q.get(True, 1)  # blocking, with 1 second timeout
            except:
                if done.is_set():  # no items in queue, and parent finished
                    break
                else:  # no items in queue, but parent might send more
                    continue

        if not test_mode:
            try:
                sock.sendall(line.encode('utf-8'))
            except:
                sock = None  # notify that we need to make a new socket at start of loop
                retry_line = line  # can't really put back in q, so remember to retry this line
                continue

        etime = time.time() - stime  #time that actually elapsed

        #Expected value of wait_time is 1/MPS_LIMIT, ie. MPS_LIMIT per second.
        if mps > 0:
            wait_time = (2.0 * random.random()) / (mps)
            if wait_time > etime:  #if we should wait
                time.sleep(wait_time - etime)  #then wait

    if sock:
        sock.close()