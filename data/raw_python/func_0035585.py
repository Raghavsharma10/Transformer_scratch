def opt_hyper(gpr,Ifilter=None,bounds=None,opts={},*args,**kw_args):
    """
    optimize params

    Input:
    gpr: GP regression class
    params0: dictionary filled with starting hyperparameters
    opts: options for optimizer
    """
    if 'gradcheck' in opts:
        gradcheck = opts['gradcheck']
    else:
        gradcheck = False
    if 'max_iter_opt' in opts:
        max_iter = opts['max_iter_opt']
    else:
        max_iter = 5000
    if 'pgtol' in opts:
        pgtol = opts['pgtol']
    else:
        pgtol = 1e-10

    params0 = gpr.getParams()

    def f(x):
        x_ = X0
        x_[Ifilter_x] = x
        gpr.setParams(param_list_to_dict(x_,param_struct,skeys))
        lml = gpr.LML()
        if SP.isnan(lml):
            lml=1E6
        lml_grad = gpr.LML_grad()
        lml_grad = param_dict_to_list(lml_grad,skeys)
        if (~SP.isfinite(lml_grad)).any():
            idx = (~SP.isfinite(lml_grad))
            lml_grad[idx] = 1E6
        return lml, lml_grad[Ifilter_x]

    skeys = SP.sort(list(params0.keys()))
    param_struct = dict([(name,params0[name].shape) for name in skeys])

    # mask params that should not be optimized
    X0 = param_dict_to_list(params0,skeys)
    if Ifilter is not None:
        Ifilter_x = SP.array(param_dict_to_list(Ifilter,skeys),dtype=bool)
    else:
        Ifilter_x = SP.ones(len(X0),dtype='bool')

    # add bounds if necessary
    if bounds is not None:
        _b = []
        for key in skeys:
            if key in list(bounds.keys()):
                _b.extend(bounds[key])
            else:
                _b.extend([[-SP.inf,+SP.inf]]*params0[key].size)
        bounds = SP.array(_b)
        bounds = bounds[Ifilter_x]

    LG.info('Starting optimization ...')
    t = time.time()

    x = X0.copy()[Ifilter_x]
    RV = optimize(f,x,maxfun=int(max_iter),pgtol=pgtol,bounds=bounds,**kw_args)
    #RVopt = optimize(f,x,messages=True,maxfun=int(max_iter),pgtol=pgtol,bounds=bounds)
    #LG.info('%s'%OPT.tnc.RCSTRINGS[RVopt[2]])
    #LG.info('Optimization is converged at iteration %d'%RVopt[1])
    #LG.info('Total time: %.2fs'%(time.time()-t))

    info = RV[2]
    conv = info['warnflag']==0

    if gradcheck:
        err = OPT.check_grad(f,df,xopt)
        LG.info("check_grad (post): %.2f"%err)

    return conv,info