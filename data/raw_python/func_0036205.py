def coda(df, window, level):
    """
    CODA processing from Windig, Phalp, & Payne 1996 Anal Chem
    """
    # pull out the data
    d = df.values

    # smooth the data and standardize it
    smooth_data = movingaverage(d, df.index, window)[0]
    stand_data = (smooth_data - smooth_data.mean()) / smooth_data.std()

    # scale the data to have unit length
    scale_data = d / np.sqrt(np.sum(d ** 2, axis=0))

    # calculate the "mass chromatographic quality" (MCQ) index
    mcq = np.sum(stand_data * scale_data, axis=0) / np.sqrt(d.shape[0] - 1)

    # filter out ions with an mcq below level
    good_ions = [i for i, q in zip(df.columns, mcq) if q >= level]
    return good_ions