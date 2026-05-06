def p_queue(p):
    """
    queue : QUEUE COLON LIFO
          | QUEUE COLON FIFO
    """
    if p[3] == "LIFO":
        p[0] = {"queue": LIFO()}

    elif p[3] == "FIFO":
        p[0] = {"queue": FIFO()}

    else:
        raise RuntimeError("Queue discipline '%s' is not supported!" % p[1])