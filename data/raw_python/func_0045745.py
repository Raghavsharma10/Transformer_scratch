def until(timeout=10, step=0.5, action=None, silent=False):
    """Yield until timeout"""
    yield

    started = time.time()
    while True:
        if action and not silent:
            log.info(action)

        if time.time() - started > timeout:
            if action and not silent:
                log.error("Timedout %s", action)
            return
        else:
            time.sleep(step)
            yield