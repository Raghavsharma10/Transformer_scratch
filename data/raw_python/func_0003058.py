def load_res_prm(lstFunc, lstFlsMsk=None):
    """Load result parameters from multiple nii files, with optional mask.

    Parameters
    ----------
    lstFunc : list,
        list of str with file names of 3D or 4D nii files
    lstFlsMsk : list, optional
        list of str with paths to 3D nii files that can act as mask/s
    Returns
    -------
    lstPrmAry : list
        The list will contain as many numpy arrays as masks were provided.
        Each array is 2D with shape [nr voxel in mask, nr nii files in lstFunc]
    objHdr : header object
        Header of nii file.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of nii data.

    """

    # load parameter/functional maps into a list
    lstPrm = []
    for ind, path in enumerate(lstFunc):
        aryFnc = load_nii(path)[0].astype(np.float32)
        if aryFnc.ndim == 3:
            lstPrm.append(aryFnc)
        # handle cases where nii array is 4D, in this case split arrays up in
        # 3D arrays and appenbd those
        elif aryFnc.ndim == 4:
            for indAx in range(aryFnc.shape[-1]):
                lstPrm.append(aryFnc[..., indAx])

    # load mask/s if available
    if lstFlsMsk is not None:
        lstMsk = [None] * len(lstFlsMsk)
        for ind, path in enumerate(lstFlsMsk):
            aryMsk = load_nii(path)[0].astype(np.bool)
            lstMsk[ind] = aryMsk
    else:
        print('------------No masks were provided')

    if lstFlsMsk is None:
        # if no mask was provided we just flatten all parameter array in list
        # and return resulting list
        lstPrmAry = [ary.flatten() for ary in lstPrm]
    else:
        # if masks are available, we loop over masks and then over parameter
        # maps to extract selected voxels and parameters
        lstPrmAry = [None] * len(lstFlsMsk)
        for indLst, aryMsk in enumerate(lstMsk):
            # prepare array that will hold parameter values of selected voxels
            aryPrmSel = np.empty((np.sum(aryMsk), len(lstPrm)),
                                 dtype=np.float32)
            # loop over different parameter maps
            for indAry, aryPrm in enumerate(lstPrm):
                # get voxels specific to this mask
                aryPrmSel[:, indAry] = aryPrm[aryMsk, ...]
            # put array away in list, if only one parameter map was provided
            # the output will be squeezed
            lstPrmAry[indLst] = aryPrmSel

    # also get header object and affine array
    # we simply take it for the first functional nii file, cause that is the
    # only file that has to be provided by necessity
    objHdr, aryAff = load_nii(lstFunc[0])[1:]

    return lstPrmAry, objHdr, aryAff