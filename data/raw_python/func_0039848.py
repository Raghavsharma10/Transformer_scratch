def recon_all(subj_id,anatomies):
    '''Run the ``recon_all`` script'''
    if not environ_setup:
        setup_freesurfer()
    if isinstance(anatomies,basestring):
        anatomies = [anatomies]
    nl.run([os.path.join(freesurfer_home,'bin','recon-all'),'-all','-subjid',subj_id] + [['-i',anat] for anat in anatomies])