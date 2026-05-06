def volreg(dset,suffix='_volreg',base=3,tshift=3,dfile_suffix='_volreg.1D'):
    '''simple interface to 3dvolreg

        :suffix:        suffix to add to ``dset`` for volreg'ed file
        :base:          either a number or ``dset[#]`` of the base image to register to
        :tshift:        if a number, then tshift ignoring that many images, if ``None``
                        then don't tshift
        :dfile_suffix:  suffix to add to ``dset`` to save the motion parameters to
    '''
    cmd = ['3dvolreg','-prefix',nl.suffix(dset,suffix),'-base',base,'-dfile',nl.prefix(dset)+dfile_suffix]
    if tshift:
        cmd += ['-tshift',tshift]
    cmd += [dset]
    nl.run(cmd,products=nl.suffix(dset,suffix))