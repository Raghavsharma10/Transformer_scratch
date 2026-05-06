def run(configobj=None):
    """TEAL interface for the `clean` function."""
    clean(configobj['input'],
          suffix=configobj['suffix'],
          stat=configobj['stat'],
          maxiter=configobj['maxiter'],
          sigrej=configobj['sigrej'],
          lower=configobj['lower'],
          upper=configobj['upper'],
          binwidth=configobj['binwidth'],
          mask1=configobj['mask1'],
          mask2=configobj['mask2'],
          dqbits=configobj['dqbits'],
          rpt_clean=configobj['rpt_clean'],
          atol=configobj['atol'],
          cte_correct=configobj['cte_correct'],
          clobber=configobj['clobber'],
          verbose=configobj['verbose'])