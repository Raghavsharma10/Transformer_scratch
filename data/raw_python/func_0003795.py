def random_unit(size=3):
    """Return a random unit vector of the given dimension

       Optional argument:
         size  --  the number of dimensions of the unit vector [default=3]
    """
    while True:
        result = np.random.normal(0, 1, size)
        length = np.linalg.norm(result)
        if length > 1e-3:
            return result/length