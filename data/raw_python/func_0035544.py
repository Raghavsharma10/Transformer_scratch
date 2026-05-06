def step_impl04(context):
    """Compare behavior of singleton vs. non-singleton.

    :param context: test context.
    """
    single = context.singleStore
    general = context.generalStore
    key = 13
    item = 42
    assert single.request(key) == general.request(key)
    single.add_item(key, item)
    general.add_item(key, item)
    assert single.request(key) == general.request(key)