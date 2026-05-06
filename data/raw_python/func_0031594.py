def sli_run(parameters=object(),
            fname='microcircuit.sli',
            verbosity='M_ERROR'):
    '''
    Takes parameter-class and name of main sli-script as input, initiating the
    simulation.
    
    kwargs:
    ::
        parameters : object, parameter class instance
        fname : str, path to sli codes to be executed
        verbosity : 'str', nest verbosity flag
    '''
    # Load parameters from params file, and pass them to nest
    # Python -> SLI
    send_nest_params_to_sli(vars(parameters))
    
    #set SLI verbosity
    nest.sli_run("%s setverbosity" % verbosity)
    
    # Run NEST/SLI simulation
    nest.sli_run('(%s) run' % fname)