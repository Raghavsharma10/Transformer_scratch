def main():
    """
    Slowly writes to stdout, without emitting a newline so any output
    buffering (or input for next pipeline command) can be detected.
    """
    now = datetime.datetime.now
    try:
        while True:
            sys.stdout.write(str(now()) + ' ')
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    except IOError as exc:
        if exc.errno != errno.EPIPE:
            raise