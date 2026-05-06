def run(configobj=None):
    """TEAL interface for :func:`destripe_plus`."""
    destripe_plus(
        configobj['input'],
        suffix=configobj['suffix'],
        stat=configobj['stat'],
        maxiter=configobj['maxiter'],
        sigrej=configobj['sigrej'],
        lower=configobj['lower'],
        upper=configobj['upper'],
        binwidth=configobj['binwidth'],
        scimask1=configobj['scimask1'],
        scimask2=configobj['scimask2'],
        dqbits=configobj['dqbits'],
        rpt_clean=configobj['rpt_clean'],
        atol=configobj['atol'],
        cte_correct=configobj['cte_correct'],
        clobber=configobj['clobber'],
        verbose=configobj['verbose'])