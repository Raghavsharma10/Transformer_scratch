def variant_stats_from_variant(variant,
                               metadata,
                               merge_fn=(lambda all_stats: \
                                max(all_stats, key=(lambda stats: stats.tumor_stats.depth)))):
    """Parse the variant calling stats from a variant called from multiple variant files. The stats are merged
    based on `merge_fn`

    Parameters
    ----------
    variant : varcode.Variant
    metadata : dict
        Dictionary of variant file to variant calling metadata from that file
    merge_fn : function
        Function from list of SomaticVariantStats to single SomaticVariantStats.
        This is used if a variant is called by multiple callers or appears in multiple VCFs.
        By default, this uses the data from the caller that had a higher tumor depth.

    Returns
    -------
    SomaticVariantStats
    """
    all_stats = []
    for (variant_file, variant_metadata) in metadata.items():
        if _vcf_is_maf(variant_file=variant_file):
            stats = maf_somatic_variant_stats(variant, variant_metadata)
        elif _vcf_is_strelka(variant_file=variant_file,
                             variant_metadata=variant_metadata):
            stats = strelka_somatic_variant_stats(variant, variant_metadata)
        elif _vcf_is_mutect(variant_file=variant_file,
                            variant_metadata=variant_metadata):
            stats = mutect_somatic_variant_stats(variant, variant_metadata)
        else:
            raise ValueError("Cannot parse sample fields, variant file {} is from an unsupported caller.".format(variant_file))
        all_stats.append(stats)
    return merge_fn(all_stats)