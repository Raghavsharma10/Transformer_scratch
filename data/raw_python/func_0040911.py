def resample_dset(dset,template,prefix=None,resam='NN'):
    '''Resamples ``dset`` to the grid of ``template`` using resampling mode ``resam``.
    Default prefix is to suffix ``_resam`` at the end of ``dset``

    Available resampling modes:
        :NN:    Nearest Neighbor
        :Li:    Linear
        :Cu:    Cubic
        :Bk:    Blocky
    '''
    if prefix==None:
        prefix = nl.suffix(dset,'_resam')
    nl.run(['3dresample','-master',template,'-rmode',resam,'-prefix',prefix,'-inset',dset])