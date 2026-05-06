def _delete_file(fileName, n=10):
    """Cleanly deletes a file in `n` attempts (if necessary)"""
    status = False
    count = 0
    while not status and count < n:
        try:
            _os.remove(fileName)
        except OSError:
            count += 1
            _time.sleep(0.2)
        else:
            status = True
    return status