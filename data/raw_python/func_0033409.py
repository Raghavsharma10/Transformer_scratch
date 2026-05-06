def batch_gen(data, batch_size):
    '''
    Usage::
        for batch in batch_gen(iter, 100):
            do_something(batch)
    '''
    data = data or []
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]