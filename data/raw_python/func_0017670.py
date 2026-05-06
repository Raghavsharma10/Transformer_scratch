def get_short_annotations(annotations):
    """
    Converts full GATK annotation name to the shortened version
    :param annotations:
    :return:
    """
    # Annotations need to match VCF header
    short_name = {'QualByDepth': 'QD',
                  'FisherStrand': 'FS',
                  'StrandOddsRatio': 'SOR',
                  'ReadPosRankSumTest': 'ReadPosRankSum',
                  'MappingQualityRankSumTest': 'MQRankSum',
                  'RMSMappingQuality': 'MQ',
                  'InbreedingCoeff': 'ID'}

    short_annotations = []
    for annotation in annotations:
        if annotation in short_name:
            annotation = short_name[annotation]
        short_annotations.append(annotation)
    return short_annotations