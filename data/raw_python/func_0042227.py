def first_from_generator(generator):
    """Pull the first value from a generator and return it, closing the generator

    :param generator:
        A generator, this will be mapped onto a list and the first item extracted.
    :return:
        None if there are no items, or the first item otherwise.
    :internal:
    """
    try:
        result = next(generator)
    except StopIteration:
        result = None
    finally:
        generator.close()
    return result