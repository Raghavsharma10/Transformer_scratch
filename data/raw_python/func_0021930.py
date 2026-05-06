def _mutect_variant_stats(variant, sample_info):
    """Parse a single sample"s variant calling statistics based on Mutect"s (v1) VCF output

    Parameters
    ----------
    variant : varcode.Variant
    sample_info : dict
        Dictionary of Mutect-specific variant calling fields

    Returns
    -------
    VariantStats
    """

    # Parse out the AD (or allele depth field), which is an array of [REF_DEPTH, ALT_DEPTH]
    ref_depth, alt_depth = sample_info["AD"]
    depth = int(ref_depth) + int(alt_depth)
    vaf = float(alt_depth) / depth

    return VariantStats(depth=depth, alt_depth=alt_depth, variant_allele_frequency=vaf)