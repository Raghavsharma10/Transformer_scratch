def addNew(label, version, baseVersion, dataHash="", fixedHash="", replayHash=""):
    """
    Add a new version record to the database to be tracked
    VERSION RECORD EXAMPLE:
        "base-version": 55505, 
        "data-hash": "60718A7CA50D0DF42987A30CF87BCB80", 
        "fixed-hash": "0189B2804E2F6BA4C4591222089E63B2", 
        "label": "3.16", 
        "replay-hash": "B11811B13F0C85C29C5D4597BD4BA5A4", 
        "version": 55505
        """
    baseVersion = int(baseVersion)
    version     = int(version)
    minVersChecks = {"base-version":baseVersion, "version":version}
    if label in handle.ALL_VERS_DATA:
        raise ValueError("given record label (%s) is already defined.  Consider performing update() for this record instead"%(label))
    for vCheckK,vCheckV in iteritems(minVersChecks): # verify no conflicting values
        maxVersion  = min([vData[vCheckK] for vData in handle.ALL_VERS_DATA.values()])
        if vCheckV < c.MIN_VERSION_AI_API:
            raise ValueError("version %s / %s.%s does not support the Starcraft2 API"%(baseVersion, label, version))
        if vCheckV < maxVersion: # base version cannot be smaller than newest value
            raise ValueError("given %s (%d) cannot be smaller than newest known %s (%d)"%(vCheckK, vCheckV, vCheckK, maxVersion))
    uniqueValHeaders = list(c.JSON_HEADERS)
    uniqueValHeaders.remove("base-version")
    record = {"base-version" : baseVersion}
    #print("%15s : %s (%s)"%("base-version", baseVersion, type(baseVersion)))
    for k,v in zip(uniqueValHeaders, [label, version, dataHash, fixedHash, replayHash]): # new attr values must be unique within all handler records
        record[k] = v # convert to dict while checking each param
        #print("%15s : %s (%s)"%(k,v,type(v)))
        if not v: continue # ignore uniqueness requirement if value is unspecified
        if v in [r[k] for r in Handler.ALL_VERS_DATA.values()]:
            raise ValueError("'%s' '%s' is in known values: %s"%(k, v, getattr(handle, k)))
            return
    handle.save(new=record)