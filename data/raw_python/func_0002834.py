def np_lst_sq(vecMdl, aryFuncChnk):
    """Least squares fitting in numpy without cross-validation.

    Notes
    -----
    This is just a wrapper function for np.linalg.lstsq to keep piping
    consistent.

    """
    aryTmpBts, vecTmpRes = np.linalg.lstsq(vecMdl,
                                           aryFuncChnk,
                                           rcond=-1)[:2]

    return aryTmpBts, vecTmpRes