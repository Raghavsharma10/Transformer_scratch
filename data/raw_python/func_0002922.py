def loadNiiData(lstNiiFls,
                strPathNiiMask=None,
                strPathNiiFunc=None):
    """load nii data.

        Parameters
        ----------
        lstNiiFls : list, list of str with nii file names
        strPathNiiMask : str, path to nii file with mask (optional)
        strPathNiiFunc : str, parent path to nii files (optional)
        Returns
        -------
        aryFunc : np.array
            Nii data   
    """
    print('---------Loading nii data')
    # check whether  a mask is available
    if strPathNiiMask is not None:
        aryMask = nb.load(strPathNiiMask).get_data().astype('bool')
    # check a parent path is available that needs to be preprended to nii files
    if strPathNiiFunc is not None:
        lstNiiFls = [os.path.join(strPathNiiFunc, i) for i in lstNiiFls]

    aryFunc = []
    for idx, path in enumerate(lstNiiFls):
        print('------------Loading run: ' + str(idx+1))
        # Load 4D nii data:
        niiFunc = nb.load(path).get_data()
        # append to list
        if strPathNiiMask is not None:
            aryFunc.append(niiFunc[aryMask, :])
        else:
            aryFunc.append(niiFunc)
    # concatenate arrys in list along time dimension
    aryFunc = np.concatenate(aryFunc, axis=-1)
    # set to type float32
    aryFunc = aryFunc.astype('float32')

    return aryFunc