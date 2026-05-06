def setup_file_dest(params, clearDestination=True):
    """
    Function to set up the file catalog structure for simulation output
    
    
    Parameters
    ----------  
    params : object 
        e.g., `cellsim16popsParams.multicompartment_params()`
    clear_dest : bool 
        Savefolder will be cleared if already existing.
    
    
    Returns
    -------
    None
    
    """
    if RANK == 0:
        if not os.path.isdir(params.savefolder):
            os.mkdir(params.savefolder)
            assert(os.path.isdir(params.savefolder))
        else:
            if clearDestination:
                print('removing folder tree %s' % params.savefolder)
                while os.path.isdir(params.savefolder):
                    try:
                        os.system('find %s -delete' % params.savefolder)
                    except:
                        shutil.rmtree(params.savefolder)
                os.mkdir(params.savefolder)
                assert(os.path.isdir(params.savefolder))
        
        if not os.path.isdir(params.sim_scripts_path):
            print('creating %s' % params.sim_scripts_path)
            os.mkdir(params.sim_scripts_path)
        
        if not os.path.isdir(params.cells_path):
            print('creating %s' % params.cells_path)
            os.mkdir(params.cells_path)
        
        if not os.path.isdir(params.figures_path):
            print('creating %s' % params.figures_path)
            os.mkdir(params.figures_path)
        
        if not os.path.isdir(params.populations_path):
            print('creating %s' % params.populations_path)
            os.mkdir(params.populations_path)
        
        try:
            if not os.path.isdir(params.raw_nest_output_path):
                print('creating %s' % params.raw_nest_output_path)
                os.mkdir(params.raw_nest_output_path)
        except:
            pass
        
        if not os.path.isdir(params.spike_output_path):
            print('creating %s' % params.spike_output_path)
            os.mkdir(params.spike_output_path)
    
        for f in ['cellsim16popsParams.py',
                  'cellsim16pops.py',
                  'example_brunel.py',
                  'brunel_alpha_nest.py',
                  'mesocircuit.sli',
                  'mesocircuit_LFP_model.py',
                  'binzegger_connectivity_table.json', 
                  'nest_simulation.py',
                  'microcircuit.sli']:
            if os.path.isfile(f):
                if not os.path.exists(os.path.join(params.sim_scripts_path, f)):
                    shutil.copy(f, os.path.join(params.sim_scripts_path, f))
                    os.chmod(os.path.join(params.sim_scripts_path, f),
                             stat.S_IREAD)
       
    COMM.Barrier()