def get_distr(center=0.0, stdev=default_stdev, length=50):
    "Returns a PDF of a given length. "
    # distr = np.random.random(length)

    # sticking to normal distibution to easily control separability
    distr = rng.normal(center, stdev, size=[length, 1])

    return distr