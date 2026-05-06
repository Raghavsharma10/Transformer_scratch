def mutect_somatic_variant_stats(variant, variant_metadata):
    """Parse out the variant calling statistics for a given variant from a Mutect VCF

    Parameters
    ----------
    variant : varcode.Variant
    sample_info : dict
        Dictionary of sample to variant calling statistics, corresponds to the sample columns
        in a Mutect VCF

    Returns
    -------
    SomaticVariantStats
    """

    sample_info = variant_metadata["sample_info"]
    # Ensure there are exactly two samples in the VCF, a tumor and normal
    assert len(sample_info) == 2, "More than two samples found in the somatic VCF"

    # Find the sample with the genotype field set to variant in the VCF
    tumor_sample_infos = [info for info in sample_info.values() if info["GT"] == "0/1"]

    # Ensure there is only one such sample
    assert len(tumor_sample_infos) == 1, "More than one tumor sample found in the VCF file"

    tumor_sample_info = tumor_sample_infos[0]
    normal_sample_info = [info for info in sample_info.values() if info["GT"] != "0/1"][0]

    tumor_stats = _mutect_variant_stats(variant, tumor_sample_info)
    normal_stats = _mutect_variant_stats(variant, normal_sample_info)
    return SomaticVariantStats(tumor_stats=tumor_stats, normal_stats=normal_stats)