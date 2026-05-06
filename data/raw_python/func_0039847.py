def setup_freesurfer():
    '''Setup the freesurfer environment variables'''
    guess_home()
    os.environ['FREESURFER_HOME'] = freesurfer_home
    os.environ['SUBJECTS_DIR'] = subjects_dir
    # Run the setup script and collect the output:
    o = subprocess.check_output(['bash','-c','source %s/SetUpFreeSurfer.sh && env' % freesurfer_home])
    env = [(a.partition('=')[0],a.partition('=')[2]) for a in o.split('\n') if len(a.strip())>0]
    for e in env:
        os.environ[e[0]] = e[1]
    environ_setup = True