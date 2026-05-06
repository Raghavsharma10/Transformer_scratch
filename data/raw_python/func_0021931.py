def maf_somatic_variant_stats(variant, variant_metadata):
    """
    Parse out the variant calling statistics for a given variant from a MAF file

    Assumes the MAF format described here: https://www.biostars.org/p/161298/#161777

    Parameters
    ----------
    variant : varcode.Variant
    variant_metadata : dict
        Dictionary of metadata for this variant

    Returns
    -------
    SomaticVariantStats
    """
    tumor_stats = None
    normal_stats = None
    if "t_ref_count" in variant_metadata:
        tumor_stats = _maf_variant_stats(variant, variant_metadata, prefix="t")
    if "n_ref_count" in variant_metadata:
        normal_stats = _maf_variant_stats(variant, variant_metadata, prefix="n")
    return SomaticVariantStats(tumor_stats=tumor_stats, normal_stats=normal_stats)