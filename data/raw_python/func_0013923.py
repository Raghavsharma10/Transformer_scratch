def median2D(const, bin1, label1, bin2, label2, data_label, 
             returnData=False):
    """Return a 2D average of data_label over a season and label1, label2.

    Parameters
    ----------
        const: Constellation or Instrument
        bin#: [min, max, number of bins]
        label#: string 
            identifies data product for bin#
        data_label: list-like 
            contains strings identifying data product(s) to be averaged

    Returns
    -------
    median : dictionary
        2D median accessed by data_label as a function of label1 and label2
        over the season delineated by bounds of passed instrument objects.
        Also includes 'count' and 'avg_abs_dev' as well as the values of
        the bin edges in 'bin_x' and 'bin_y'.
    
    """

    # const is either an Instrument or a Constellation, and we want to 
    #  iterate over it. 
    # If it's a Constellation, then we can do that as is, but if it's
    #  an Instrument, we just have to put that Instrument into something
    #  that will yeild that Instrument, like a list.
    if isinstance(const, pysat.Instrument):
        const = [const]
    elif not isinstance(const, pysat.Constellation):
        raise ValueError("Parameter must be an Instrument or a Constellation.")

    # create bins
    #// seems to create the boundaries used for sorting into bins
    binx = np.linspace(bin1[0], bin1[1], bin1[2]+1)
    biny = np.linspace(bin2[0], bin2[1], bin2[2]+1)

    #// how many bins are used
    numx = len(binx)-1
    numy = len(biny)-1
    #// how many different data products
    numz = len(data_label)

    # create array to store all values before taking median
    #// the indices of the bins/data products? used for looping.
    yarr = np.arange(numy)
    xarr = np.arange(numx)
    zarr = np.arange(numz)
    #// 3d array:  stores the data that is sorted into each bin? - in a deque
    ans = [ [ [collections.deque() for i in xarr] for j in yarr] for k in zarr]

    for inst in const:
        # do loop to iterate over instrument season
        #// probably iterates by date but that all depends on the
        #// configuration of that particular instrument. 
        #// either way, it iterates over the instrument, loading successive
        #// data between start and end bounds
        for inst in inst:
            # collect data in bins for averaging
            if len(inst.data) != 0:
                #// sort the data into bins (x) based on label 1
                #// (stores bin indexes in xind)
                xind = np.digitize(inst.data[label1], binx)-1
                #// for each possible x index
                for xi in xarr:
                    #// get the indicies of those pieces of data in that bin
                    xindex, = np.where(xind==xi)
                    if len(xindex) > 0:
                        #// look up the data along y (label2) at that set of indicies (a given x)
                        yData = inst.data.iloc[xindex]
                        #// digitize that, to sort data into bins along y (label2) (get bin indexes)
                        yind = np.digitize(yData[label2], biny)-1
                        #// for each possible y index
                        for yj in yarr:
                            #// select data with this y index (and we already filtered for this x index)
                            yindex, = np.where(yind==yj)
                            if len(yindex) > 0:
                                #// for each data product label zk
                                for zk in zarr:
                                    #// take the data (already filtered by x); filter it by y and 
                                    #// select the data product, put it in a list, and extend the deque
                                    ans[zk][yj][xi].extend( yData.ix[yindex,data_label[zk]].tolist() )

    return _calc_2d_median(ans, data_label, binx, biny, xarr, yarr, zarr, numx, numy, numz, returnData)