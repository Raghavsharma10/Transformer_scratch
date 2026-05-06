def distribute(total, objects):
    '''Generator for (count, object) tuples that distributes count evenly among
    the provided objects'''
    for index, obj in enumerate(objects):
        start = (index * total) / len(objects)
        stop = ((index + 1) * total) / len(objects)
        yield (stop - start, obj)