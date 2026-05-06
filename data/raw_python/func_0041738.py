def auto_qc(dset,inside_perc=60,atlas=None,p=0.001):
    '''returns ``False`` if ``dset`` fails minimum checks, or returns a float from ``0.0`` to ``100.0`` describing data quality'''
    with nl.notify('Running quality check on %s:' % dset):
        if not os.path.exists(dset):
            nl.notify('Error: cannot find the file!',level=nl.level.error)
            return False
        
        info = nl.dset_info(dset)
        if not info:
            nl.notify('Error: could not read the dataset!',level=nl.level.error)
        
        if any(['stat' in x for x in info.subbricks]):
            with nl.notify('Statistical results detected...'):
                inside = inside_brain(dset,atlas=atlas,p=p)
                nl.notify('%.1f significant voxels inside brain')
                if inside<inside_perc:
                    nl.notify('Warning: below quality threshold!',level=nl.level.warning)
#                    return False
                nl.notify('Looks ok')
                return inside
        
        if len(info.subbricks)>1:
            with nl.notify('Time-series detected...'):
                return_val = True
                (cost,overlap) = atlas_overlap(dset)
                if cost>0.15 or overlap<80:
                    nl.notify('Warning: does not appear to conform to brain dimensions',level=nl.level.warning)
                    return_val = False
                if len(info.subbricks)>5:
                    (oc,perc_outliers) = outcount(dset)
                    if perc_outliers>0.1:
                        nl.notify('Warning: large amount of outlier time points',level=nl.level.warning)
                        return_val = False
            if return_val:
                nl.notify('Looks ok')
                return min(100*(1-cost),overlap,100*perc_outliers)
            return False
        
        with nl.notify('Single brain image detected...'):
            (cost,overlap) = atlas_overlap(dset)
            # Be more lenient if it's not an EPI, cuz who knows what else is in this image
            if cost>0.45 or overlap<70:
                nl.notify('Warning: does not appear to conform to brain dimensions',level=nl.level.warning)
                return False
            nl.notify('Looks ok')
            return min(100*(1-cost),overlap)