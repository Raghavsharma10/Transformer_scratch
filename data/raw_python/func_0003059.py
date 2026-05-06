def export_nii(ary2dNii, lstNiiNames, aryLgcMsk, aryLgcVar, tplNiiShp, aryAff,
               hdrMsk, outFormat='3D'):
    """
    Export nii file(s).

    Parameters
    ----------
    ary2dNii : numpy array
        Numpy array with results to be exported to nii.
    lstNiiNames : list
        List that contains strings with the complete file names.
    aryLgcMsk : numpy array
        If the nii file is larger than this threshold (in MB), the file is
        loaded volume-by-volume in order to prevent memory overflow. Default
        threshold is 1000 MB.
    aryLgcVar : np.array
        1D numpy array containing logical values. One value per voxel after
        mask has been applied. If `True`, the variance and mean of the voxel's
        time course are greater than the provided thresholds in all runs and
        the voxel is included in the output array (`aryFunc`). If `False`, the
        variance or mean of the voxel's time course is lower than threshold in
        at least one run and the voxel has been excluded from the output
        (`aryFunc`). This is to avoid problems in the subsequent model fitting.
        This array is necessary to put results into original dimensions after
        model fitting.
    tplNiiShp : tuple
        Tuple that describes the 3D shape of the output volume
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of nii data.
    hdrMsk : nibabel-header-object
        Nii header of mask.
    outFormat : string, either '3D' or '4D'
        String specifying whether images will be saved as seperate 3D nii
        files or one 4D nii file

    Notes
    -----
    [1] This function does not return any arrays but instead saves to disk.
    [2] Depending on whether outFormat is '3D' or '4D' images will be saved as
        seperate 3D nii files or one 4D nii file.
    """

    # Number of voxels that were included in the mask:
    varNumVoxMsk = np.sum(aryLgcMsk)

    # Number of maps in ary2dNii
    varNumMaps = ary2dNii.shape[-1]

    # Place voxels based on low-variance exlusion:
    aryPrfRes01 = np.zeros((varNumVoxMsk, varNumMaps), dtype=np.float32)
    for indMap in range(varNumMaps):
        aryPrfRes01[aryLgcVar, indMap] = ary2dNii[:, indMap]

    # Total number of voxels:
    varNumVoxTlt = (tplNiiShp[0] * tplNiiShp[1] * tplNiiShp[2])

    # Place voxels based on mask-exclusion:
    aryPrfRes02 = np.zeros((varNumVoxTlt, aryPrfRes01.shape[-1]),
                           dtype=np.float32)
    for indDim in range(aryPrfRes01.shape[-1]):
        aryPrfRes02[aryLgcMsk, indDim] = aryPrfRes01[:, indDim]

    # Reshape pRF finding results into original image dimensions:
    aryPrfRes = np.reshape(aryPrfRes02,
                           [tplNiiShp[0],
                            tplNiiShp[1],
                            tplNiiShp[2],
                            aryPrfRes01.shape[-1]])

    if outFormat == '3D':
        # Save nii results:
        for idxOut in range(0, aryPrfRes.shape[-1]):
            # Create nii object for results:
            niiOut = nb.Nifti1Image(aryPrfRes[..., idxOut],
                                    aryAff,
                                    header=hdrMsk
                                    )
            # Save nii:
            strTmp = lstNiiNames[idxOut]
            nb.save(niiOut, strTmp)

    elif outFormat == '4D':

        # adjust header
        hdrMsk.set_data_shape(aryPrfRes.shape)

        # Create nii object for results:
        niiOut = nb.Nifti1Image(aryPrfRes,
                                aryAff,
                                header=hdrMsk
                                )
        # Save nii:
        strTmp = lstNiiNames[0]
        nb.save(niiOut, strTmp)