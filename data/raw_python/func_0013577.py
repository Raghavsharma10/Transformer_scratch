def rnaseq2ga(quantificationFilename, sqlFilename, localName, rnaType,
              dataset=None, featureType="gene",
              description="", programs="", featureSetNames="",
              readGroupSetNames="", biosampleId=""):
    """
    Reads RNA Quantification data in one of several formats and stores the data
    in a sqlite database for use by the GA4GH reference server.

    Supports the following quantification output types:
    Cufflinks, kallisto, RSEM.
    """
    readGroupSetName = ""
    if readGroupSetNames:
        readGroupSetName = readGroupSetNames.strip().split(",")[0]
    featureSetIds = ""
    readGroupIds = ""
    if dataset:
        featureSetIdList = []
        if featureSetNames:
            for annotationName in featureSetNames.split(","):
                featureSet = dataset.getFeatureSetByName(annotationName)
                featureSetIdList.append(featureSet.getId())
            featureSetIds = ",".join(featureSetIdList)
        # TODO: multiple readGroupSets
        if readGroupSetName:
            readGroupSet = dataset.getReadGroupSetByName(readGroupSetName)
            readGroupIds = ",".join(
                [x.getId() for x in readGroupSet.getReadGroups()])
    if rnaType not in SUPPORTED_RNA_INPUT_FORMATS:
        raise exceptions.UnsupportedFormatException(rnaType)
    rnaDB = RnaSqliteStore(sqlFilename)
    if rnaType == "cufflinks":
        writer = CufflinksWriter(rnaDB, featureType, dataset=dataset)
    elif rnaType == "kallisto":
        writer = KallistoWriter(rnaDB, featureType, dataset=dataset)
    elif rnaType == "rsem":
        writer = RsemWriter(rnaDB, featureType, dataset=dataset)
    writeRnaseqTable(rnaDB, [localName], description, featureSetIds,
                     readGroupId=readGroupIds, programs=programs,
                     biosampleId=biosampleId)
    writeExpressionTable(writer, [(localName, quantificationFilename)])