def create_dset_to3d(prefix,file_list,file_order='zt',num_slices=None,num_reps=None,TR=None,slice_order='alt+z',only_dicoms=True,sort_filenames=False):
    '''manually create dataset by specifying everything (not recommended, but necessary when autocreation fails)

    If `num_slices` or `num_reps` is omitted, it will be inferred by the number of images. If both are omitted,
    it assumes that this it not a time-dependent dataset

    :only_dicoms:       filter the given list by readable DICOM images
    :sort_filenames:    sort the given files by filename using the right-most number in the filename'''

    tags = {
        'num_rows': (0x0028,0x0010),
        'num_reps': (0x0020,0x0105),
        'TR': (0x0018,0x0080)
    }
    with nl.notify('Trying to create dataset %s' % prefix):
        if os.path.exists(prefix):
            nl.notify('Error: file "%s" already exists!' % prefix,level=nl.level.error)
            return False

        tagvals = {}
        for f in file_list:
            try:
                tagvals[f] = info_for_tags(f,tags.values())
            except:
                pass
        if only_dicoms:
            new_file_list = []
            for f in file_list:
                if f in tagvals and len(tagvals[f][tags['num_rows']])>0:
                    # Only include DICOMs that actually have image information
                    new_file_list.append(f)
            file_list = new_file_list

        if sort_filenames:
            def file_num(fname):
                try:
                    nums = [x.strip('.') for x in re.findall(r'[\d.]+',fname) if x.strip('.')!='']
                    return float(nums[-1])
                except:
                    return fname
            file_list = sorted(file_list,key=file_num)

        if len(file_list)==0:
            nl.notify('Error: Couldn\'t find any valid DICOM images',level=nl.level.error)
            return False


        cmd = ['to3d','-skip_outliers','-quit_on_err','-prefix',prefix]

        if num_slices!=None or num_reps!=None:
            # Time-based dataset
            if num_slices==None:
                if len(file_list)%num_reps!=0:
                    nl.notify('Error: trying to guess # of slices, but %d (number for files) doesn\'t divide evenly into %d (number of reps)' % (len(file_list),num_reps),level=nl.level.error)
                    return False
                num_slices = len(file_list)/num_reps
            if num_reps==None:
                if len(file_list)%num_slices==0:
                    num_reps = len(file_list)/num_slices
                elif len(file_list)==1 and tags['num_reps'] in tagvals[file_list[0]]:
                    num_reps = tagvals[file_list[0]][tags['num_reps']]
                else:
                    nl.notify('Error: trying to guess # of reps, but %d (number for files) doesn\'t divide evenly into %d (number of slices)' % (len(file_list),num_slices),level=nl.level.error)
                    return False

            if TR==None:
                TR = tagvals[file_list[0]][tags['TR']]
            cmd += ['-time:%s'%file_order]
            if file_order=='zt':
                cmd += [num_slices,num_reps]
            else:
                cmd += [num_reps,num_slices]
            cmd += [TR,slice_order]
        cmd += ['-@']
        cmd = [str(x) for x in cmd]
        out = None
        try:
            p = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            out = p.communicate('\n'.join(file_list))
            if p.returncode!=0:
                raise Exception
        except:
            with nl.notify('Error: to3d returned error',level=nl.level.error):
                if out:
                    nl.notify('stdout:\n' + out[0] + '\nstderr:\n' + out[1],level=nl.level.error)
            return False