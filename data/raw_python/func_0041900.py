def skullstrip_template(dset,template,prefix=None,suffix=None,dilate=0):
    '''Takes the raw anatomy ``dset``, aligns it to a template brain, and applies a templated skullstrip. Should produce fairly reliable skullstrips as long
    as there is a decent amount of normal brain and the overall shape of the brain is normal-ish'''
    if suffix==None:
        suffix = '_sstemplate'
    if prefix==None:
        prefix = nl.suffix(dset,suffix)
    if not os.path.exists(prefix):
        with nl.notify('Running template-based skull-strip on %s' % dset):
            dset = os.path.abspath(dset)
            template = os.path.abspath(template)
            tmp_dir = tempfile.mkdtemp()
            cwd = os.getcwd()
            with nl.run_in(tmp_dir):
                nl.affine_align(template,dset,skull_strip=None,cost='mi',opts=['-nmatch','100%'])
                nl.run(['3dQwarp','-minpatch','20','-penfac','10','-noweight','-source',nl.suffix(template,'_aff'),'-base',dset,'-prefix',nl.suffix(template,'_qwarp')],products=nl.suffix(template,'_qwarp'))
                info = nl.dset_info(nl.suffix(template,'_qwarp'))
                max_value = info.subbricks[0]['max']
                nl.calc([dset,nl.suffix(template,'_qwarp')],'a*step(b-%f*0.05)'%max_value,prefix)
                shutil.move(prefix,cwd)
            shutil.rmtree(tmp_dir)