def index(index, length):
    """Generates an index.

    :param index: The index, can be positive or negative.
    :param length: The length of the sequence to index.

    :raises: IndexError

    Negative indices are typically used to index a sequence in reverse order.
    But to use them, the indexed object must convert them to the correct,
    positive index. This function can be used to do this.

    """
    if index < 0:
        index += length
    if 0 <= index < length:
        return index
    raise IndexError()