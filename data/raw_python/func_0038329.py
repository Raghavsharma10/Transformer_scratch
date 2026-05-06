def adapt_array(arr):
    """
    Adapts a Numpy array into an ARRAY string to put into the database.

    Parameters
    ----------
    arr: array
        The Numpy array to be adapted into an ARRAY type that can be inserted into a SQL file.

    Returns
    -------
    ARRAY
            The adapted array object

    """
    out = io.BytesIO()
    np.save(out, arr), out.seek(0)
    return buffer(out.read())