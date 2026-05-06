def get_cur_mem_use():
    """return utilization of memory"""
    # http://lwn.net/Articles/28345/

    lines = open("/proc/meminfo", 'r').readlines()
    emptySpace = re.compile('[ ]+')
    for line in lines:
        if "MemTotal" in line:
            memtotal = float(emptySpace.split(line)[1])
        if "SwapFree" in line:
            swapfree = float(emptySpace.split(line)[1])
        if "SwapTotal" in line:
            swaptotal = float(emptySpace.split(line)[1])
        if "MemFree" in line:
            memfree = float(emptySpace.split(line)[1])
        if "Cached" in line and not "SwapCached" in line:
            cached = float(emptySpace.split(line)[1])

    ramoccup = 1.0 - (memfree + cached) / memtotal
    if swaptotal == 0:
        swapoccup = 0
    else:
        swapoccup = 1.0 - swapfree / swaptotal
    strramoccup = str(round(ramoccup * 100.0, 1))
    strswapoccup = str(round(swapoccup * 100.0, 1))

    return float(memtotal), strramoccup, strswapoccup