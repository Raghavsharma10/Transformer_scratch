def smooth_decon_to_fwhm(decon,fwhm,cache=True):
    '''takes an input :class:`Decon` object and uses ``3dBlurToFWHM`` to make the output as close as possible to ``fwhm``
    returns the final measured fwhm. If ``cache`` is ``True``, will save the blurred input file (and use it again in the future)'''
    if os.path.exists(decon.prefix):
        return
    blur_dset = lambda dset: nl.suffix(dset,'_smooth_to_%.2f' % fwhm)

    with nl.notify('Running smooth_decon_to_fwhm analysis (with %.2fmm blur)' % fwhm):
        tmpdir = tempfile.mkdtemp()
        try:
            cwd = os.getcwd()
            random_files = [re.sub(r'\[\d+\]$','',str(x)) for x in nl.flatten([x for x in decon.__dict__.values() if isinstance(x,basestring) or isinstance(x,list)]+[x.values() for x in decon.__dict__.values() if isinstance(x,dict)])]
            files_to_copy = [x for x in random_files if os.path.exists(x) and x[0]!='/']
            files_to_copy += [blur_dset(dset) for dset in decon.input_dsets if os.path.exists(blur_dset(dset))]
            # copy crap
            for file in files_to_copy:
                try:
                    shutil.copytree(file,tmpdir)
                except OSError as e:
                    shutil.copy(file,tmpdir)
                shutil.copy(file,tmpdir)

            copyback_files = [decon.prefix,decon.errts]
            with nl.run_in(tmpdir):
                if os.path.exists(decon.prefix):
                    os.remove(decon.prefix)

                # Create the blurred inputs (or load from cache)
                if cache and all([os.path.exists(os.path.join(cwd,blur_dset(dset))) for dset in decon.input_dsets]):
                    # Everything is already cached...
                    nl.notify('Using cache\'d blurred datasets')
                else:
                    # Need to make them from scratch
                    with nl.notify('Creating blurred datasets'):
                        old_errts = decon.errts
                        decon.errts = 'residual.nii.gz'
                        decon.prefix = os.path.basename(decon.prefix)
                        # Run once in place to get the residual dataset
                        decon.run()
                        running_reps = 0
                        for dset in decon.input_dsets:
                            info = nl.dset_info(dset)
                            residual_dset = nl.suffix(dset,'_residual')
                            nl.run(['3dbucket','-prefix',residual_dset,'%s[%d..%d]'%(decon.errts,running_reps,running_reps+info.reps-1)],products=residual_dset)
                            cmd = ['3dBlurToFWHM','-quiet','-input',dset,'-blurmaster',residual_dset,'-prefix',blur_dset(dset),'-FWHM',fwhm]
                            if decon.mask:
                                if decon.mask=='auto':
                                    cmd += ['-automask']
                                else:
                                    cmd += ['-mask',decon.mask]
                            nl.run(cmd,products=blur_dset(dset))
                            running_reps += info.reps
                            if cache:
                                copyback_files.append(blur_dset(dset))
                    decon.errts = old_errts
                decon.input_dsets = [blur_dset(dset) for dset in decon.input_dsets]
                for d in [decon.prefix,decon.errts]:
                    if os.path.exists(d):
                        try:
                            os.remove(d)
                        except:
                            pass
                decon.run()
                for copyfile in copyback_files:
                    if os.path.exists(copyfile):
                        shutil.copy(copyfile,cwd)
                    else:
                        nl.notify('Warning: deconvolve did not produce expected file %s' % decon.prefix,level=nl.level.warning)
        except:
            raise
        finally:
            shutil.rmtree(tmpdir,True)