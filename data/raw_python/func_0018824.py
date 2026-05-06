def await_server(port, seconds):
    """Block the current process until either the *HydPy* server is responding
    on the given `port` or the given number of `seconds` elapsed.

    >>> from hydpy import run_subprocess, TestIO
    >>> with TestIO():    # doctest: +ELLIPSIS
    ...     run_subprocess('hyd.py await_server 8080 0.1')
    Invoking hyd.py with arguments `...hyd.py, await_server, 8080, 0.1` \
resulted in the following error:
    <urlopen error Waited for 0.1 seconds without response on port 8080.>
    ...

    >>> from hydpy.core.examples import prepare_full_example_1
    >>> prepare_full_example_1()
    >>> with TestIO():
    ...     process = run_subprocess(
    ...         'hyd.py start_server 8080 LahnH multiple_runs.xml',
    ...         blocking=False, verbose=False)
    ...     run_subprocess('hyd.py await_server 8080 10', verbose=False)

    >>> from urllib import request
    >>> _ = request.urlopen('http://localhost:8080/close_server')
    >>> process.kill()
    >>> _ = process.communicate()
    """
    now = time.perf_counter()
    end = now + float(seconds)
    while now <= end:
        try:
            urllib.request.urlopen(f'http://localhost:{port}/status')
            break
        except urllib.error.URLError:
            time.sleep(0.1)
            now = time.perf_counter()
    else:
        raise urllib.error.URLError(
            f'Waited for {seconds} seconds without response on port {port}.')