def get_max_size(pool, num_option, item_length):
    """
    Calculate the max number of item that an option can stored in the pool at give time.

    This is to limit the pool size to POOL_SIZE

    Args:
        option_index (int): the index of the option to calculate the size for
        pool (dict): answer pool
        num_option (int): total number of options available for the question
        item_length (int): the length of the item

    Returns:
        int: the max number of items that `option_index` can have
    """
    max_items = POOL_SIZE / item_length
    # existing items plus the reserved for min size. If there is an option has 1 item, POOL_OPTION_MIN_SIZE - 1 space
    # is reserved.
    existing = POOL_OPTION_MIN_SIZE * num_option + sum([max(0, len(pool.get(i, {})) - 5) for i in xrange(num_option)])
    return int(max_items - existing)