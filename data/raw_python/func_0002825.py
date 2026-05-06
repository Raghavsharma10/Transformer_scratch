def crt_mdl_rsp(arySptExpInf, tplPngSize, aryMdlParams, varPar, strCrd='crt',
                lgcPrint=True):
    """Create responses of 2D Gauss models to spatial conditions.

    Parameters
    ----------
    arySptExpInf : 3d numpy array, shape [n_x_pix, n_y_pix, n_conditions]
        All spatial conditions stacked along second axis.
    tplPngSize : tuple, 2
        Pixel dimensions of the visual space (width, height).
    aryMdlParams : 2d numpy array, shape [n_x_pos*n_y_pos*n_sd, 3]
        Model parameters (x, y, sigma) for all models.
    varPar : int, positive
        Number of cores to parallelize over.
    strCrd, string, either 'crt' or 'pol'
        Whether model parameters are provided in cartesian or polar coordinates
    lgcPrint : boolean
        Whether print statements should be executed.

    Returns
    -------
    aryMdlCndRsp : 2d numpy array, shape [n_x_pos*n_y_pos*n_sd, n_cond]
        Responses of 2D Gauss models to spatial conditions.

    """

    if varPar == 1:
        # if the number of cores requested by the user is equal to 1,
        # we save the overhead of multiprocessing by calling aryMdlCndRsp
        # directly
        aryMdlCndRsp = cnvl_2D_gauss(0, aryMdlParams, arySptExpInf,
                                     tplPngSize, None, strCrd=strCrd)

    else:

        # The long array with all the combinations of model parameters is put
        # into separate chunks for parallelisation, using a list of arrays.
        lstMdlParams = np.array_split(aryMdlParams, varPar)

        # Create a queue to put the results in:
        queOut = mp.Queue()

        # Empty list for results from parallel processes (for pRF model
        # responses):
        lstMdlTc = [None] * varPar

        # Empty list for processes:
        lstPrcs = [None] * varPar
        
        if lgcPrint:
            print('---------Running parallel processes')

        # Create processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc] = mp.Process(target=cnvl_2D_gauss,
                                         args=(idxPrc, lstMdlParams[idxPrc],
                                               arySptExpInf, tplPngSize, queOut
                                               ),
                                         kwargs={'strCrd': strCrd},
                                         )
            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

        # Start processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc].start()

        # Collect results from queue:
        for idxPrc in range(0, varPar):
            lstMdlTc[idxPrc] = queOut.get(True)

        # Join processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc].join()

        if lgcPrint:
            print('---------Collecting results from parallel processes')
        # Put output arrays from parallel process into one big array
        lstMdlTc = sorted(lstMdlTc)
        aryMdlCndRsp = np.empty((0, arySptExpInf.shape[-1]))
        for idx in range(0, varPar):
            aryMdlCndRsp = np.concatenate((aryMdlCndRsp, lstMdlTc[idx][1]),
                                          axis=0)

        # Clean up:
        del(lstMdlParams)
        del(lstMdlTc)

    return aryMdlCndRsp.astype('float16')