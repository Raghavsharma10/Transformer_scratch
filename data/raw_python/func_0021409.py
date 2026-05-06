def contains_variance(arrays, names):
    """
    Make sure both arrays for bivariate ("scatter") plot have a stddev > 0
    """
    for ar, name in zip(arrays, names):
        if np.std(ar) == 0:
            sys.stderr.write(
                "No variation in '{}', skipping bivariate plots.\n".format(name.lower()))
            logging.info("Nanoplotter: No variation in {}, skipping bivariate plot".format(name))
            return False
    else:
        return True