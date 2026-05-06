def strelka_somatic_variant_stats(variant, variant_metadata):
    """Parse out the variant calling statistics for a given variant from a Strelka VCF

    Parameters
    ----------
    variant : varcode.Variant
    sample_info : dict
        Dictionary of sample to variant calling statistics, corresponds to the sample columns
        in a Strelka VCF

    Returns
    -------
    SomaticVariantStats
    """

    sample_info = variant_metadata["sample_info"]
    # Ensure there are exactly two samples in the VCF, a tumor and normal
    assert len(sample_info) == 2, "More than two samples found in the somatic VCF"
    tumor_stats = _strelka_variant_stats(variant, sample_info["TUMOR"])
    normal_stats = _strelka_variant_stats(variant, sample_info["NORMAL"])
    return SomaticVariantStats(tumor_stats=tumor_stats, normal_stats=normal_stats)