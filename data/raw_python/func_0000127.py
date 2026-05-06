def getcosmo(cosmology):
    """ Find cosmological parameters for named cosmo in cosmology.py list """

    defaultcosmologies = {'dragons': cg.DRAGONS(), 'wmap1': cg.WMAP1_Mill(),
                          'wmap3': cg.WMAP3_ML(), 'wmap5': cg.WMAP5_mean(),
                          'wmap7': cg.WMAP7_ML(), 'wmap9': cg.WMAP9_ML(),
                          'wmap1_lss': cg.WMAP1_2dF_mean(),
                          'wmap3_mean': cg.WMAP3_mean(),
                          'wmap5_ml': cg.WMAP5_ML(),
                          'wmap5_lss': cg.WMAP5_BAO_SN_mean(),
                          'wmap7_lss': cg.WMAP7_BAO_H0_mean(),
                          'planck13': cg.Planck_2013(),
                          'planck15': cg.Planck_2015()}

    if isinstance(cosmology, dict):
        # User providing their own variables
        cosmo = cosmology
        if 'A_scaling' not in cosmology.keys():
            A_scaling = getAscaling(cosmology, newcosmo=True)
            cosmo.update({'A_scaling': A_scaling})

        # Add extra variables by hand that cosmolopy requires
        # note that they aren't used (set to zero)
        for paramnames in cg.WMAP5_mean().keys():
            if paramnames not in cosmology.keys():
                cosmo.update({paramnames: 0})
    elif cosmology.lower() in defaultcosmologies.keys():
        # Load by name of cosmology instead
        cosmo = defaultcosmologies[cosmology.lower()]
        A_scaling = getAscaling(cosmology)
        cosmo.update({'A_scaling': A_scaling})
    else:
        print("You haven't passed a dict of cosmological parameters ")
        print("OR a recognised cosmology, you gave %s" % (cosmology))
    # No idea why this has to be done by hand but should be O_k = 0
    cosmo = cp.distance.set_omega_k_0(cosmo)

    # Use the cosmology as **cosmo passed to cosmolopy routines
    return(cosmo)