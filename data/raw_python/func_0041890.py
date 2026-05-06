def motion_from_params(param_file,motion_file,individual=True,rms=True):
    '''calculate a motion regressor from the params file given by 3dAllineate

    Basically just calculates the rms change in the translation and rotation components. Returns the 6 motion vector (if ``individual`` is ``True``) and the RMS difference (if ``rms`` is ``True``).'''
    with open(param_file) as inf:
        translate_rotate = np.array([[float(y) for y in x.strip().split()[:6]] for x in inf.readlines() if x[0]!='#'])
        motion = np.array([])
        if individual:
            motion = np.vstack((np.zeros(translate_rotate.shape[1]),np.diff(translate_rotate,axis=0)))
        if rms:
            translate = [sqrt(sum([x**2 for x in y[:3]])) for y in translate_rotate]
            rotate = [sqrt(sum([x**2 for x in y[3:]])) for y in translate_rotate]
            translate_rotate = np.array(map(add,translate,rotate))
            translate_rotate_diff = np.hstack(([0],np.diff(translate_rotate,axis=0)))
            if motion.shape==(0,):
                motion = rms_motion
            else:
                motion = np.column_stack((motion,translate_rotate_diff))
        with open(motion_file,'w') as outf:
            outf.write('\n'.join(['\t'.join([str(y) for y in x]) for x in motion]))