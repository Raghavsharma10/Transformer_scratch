def _strelka_variant_stats(variant, sample_info):
    """Parse a single sample"s variant calling statistics based on Strelka VCF output

    Parameters
    ----------
    variant : varcode.Variant
    sample_info : dict
        Dictionary of Strelka-specific variant calling fields

    Returns
    -------
    VariantStats
    """
    
    if variant.is_deletion or variant.is_insertion:
        # ref: https://sites.google.com/site/strelkasomaticvariantcaller/home/somatic-variant-output
        ref_depth = int(sample_info['TAR'][0]) # number of reads supporting ref allele (non-deletion)
        alt_depth = int(sample_info['TIR'][0]) # number of reads supporting alt allele (deletion)
        depth = ref_depth + alt_depth
    else:
        # Retrieve the Tier 1 counts from Strelka
        ref_depth = int(sample_info[variant.ref+"U"][0])
        alt_depth = int(sample_info[variant.alt+"U"][0])
        depth = alt_depth + ref_depth
    if depth > 0:
        vaf = float(alt_depth) / depth
    else:
        # unclear how to define vaf if no reads support variant
        # up to user to interpret this (hopefully filtered out in QC settings)
        vaf = None

    return VariantStats(depth=depth, alt_depth=alt_depth, variant_allele_frequency=vaf)