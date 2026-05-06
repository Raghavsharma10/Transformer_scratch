def outcount(dset,fraction=0.1):
    '''gets outlier count and returns ``(list of proportion of outliers by timepoint,total percentage of outlier time points)'''
    polort = nl.auto_polort(dset)
    info = nl.dset_info(dset)
    o = nl.run(['3dToutcount','-fraction','-automask','-polort',polort,dset],stderr=None,quiet=None)
    if o.return_code==0 and o.output:
        oc = [float(x) for x in o.output.split('\n') if x.strip()!='']
        binary_outcount = [x<fraction for x in oc]
        perc_outliers = 1 - (sum(binary_outcount)/float(info.reps))
        return (oc,perc_outliers)