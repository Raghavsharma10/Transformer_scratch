def get_pip(mov=None, api=None, name=None):
    """get value of pip"""
    # ~ check args
    if mov is None and api is None:
        logger.error("need at least one of those")
        raise ValueError()
    elif mov is not None and api is not None:
        logger.error("mov and api are exclusive")
        raise ValueError()
    if api is not None:
        if name is None:
            logger.error("need a name")
            raise ValueError()
        mov = api.new_mov(name)
        mov.open()
    if mov is not None:
        mov._check_open()
    # find in the collection
    try:
        logger.debug(len(Glob().theCollector.collection))
        pip = Glob().theCollector.collection['pip']
        if name is not None:
            pip_res = pip[name]
        elif mov is not None:
            pip_res = pip[mov.product]
        logger.debug("pip found in the collection")
        return pip_res
    except KeyError:
        logger.debug("pip not found in the collection")
    # ~ vars
    records = []
    intervals = [10, 20, 30]

    def _check_price(interval=10):
        timeout = time.time() + interval
        while time.time() < timeout:
            records.append(mov.get_price())
            time.sleep(0.5)

    # find variation
    for interval in intervals:
        _check_price(interval)
        if min(records) == max(records):
            logger.debug("no variation in %d seconds" % interval)
            if interval == intervals[-1]:
                raise TimeoutError("no variation")
        else:
            break
    # find longer price
    for price in records:
        if 'best_price' not in locals():
            best_price = price
        if len(str(price)) > len(str(best_price)):
            logger.debug("found new best_price %f" % price)
            best_price = price
    # get pip
    pip = get_number_unit(best_price)
    Glob().pipHandler.add_val({mov.product: pip})
    return pip