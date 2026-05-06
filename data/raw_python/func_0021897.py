def median_vaf_purity(row, cohort, **kwargs):
    """
    Estimate purity based on 2 * median VAF.

    Even if the Cohort has a default filter_fn, ignore it: we want to use all variants for
    this estimate.
    """
    patient_id = row["patient_id"]
    patient = cohort.patient_from_id(patient_id)
    variants = cohort.load_variants(patients=[patient], filter_fn=no_filter)
    if patient_id in variants.keys():
        variants = variants[patient_id]
    else:
        return np.nan
    def grab_vaf(variant):
        filterable_variant = FilterableVariant(variant, variants, patient)
        return variant_stats_from_variant(variant, filterable_variant.variant_metadata).tumor_stats.variant_allele_frequency
    vafs = [grab_vaf(variant) for variant in variants]
    return 2 * pd.Series(vafs).median()