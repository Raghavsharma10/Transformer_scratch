def temporal_snr(signal_dset,noise_dset,mask=None,prefix='temporal_snr.nii.gz'):
    '''Calculates temporal SNR by dividing average signal of ``signal_dset`` by SD of ``noise_dset``.
    ``signal_dset`` should be a dataset that contains the average signal value (i.e., nothing that has
    been detrended by removing the mean), and ``noise_dset`` should be a dataset that has all possible
    known signal fluctuations (e.g., task-related effects) removed from it (the residual dataset from a 
    deconvolve works well)'''
    for d in [('mean',signal_dset), ('stdev',noise_dset)]:
        new_d = nl.suffix(d[1],'_%s' % d[0])
        cmd = ['3dTstat','-%s' % d[0],'-prefix',new_d]
        if mask:
            cmd += ['-mask',mask]
        cmd += [d[1]]
        nl.run(cmd,products=new_d)
    nl.calc([nl.suffix(signal_dset,'_mean'),nl.suffix(noise_dset,'_stdev')],'a/b',prefix=prefix)