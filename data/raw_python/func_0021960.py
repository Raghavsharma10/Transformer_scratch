def filter_variants(variant_collection, patient, filter_fn, **kwargs):
    """Filter variants from the Variant Collection

    Parameters
    ----------
    variant_collection : varcode.VariantCollection
    patient : cohorts.Patient
    filter_fn: function
        Takes a FilterableVariant and returns a boolean. Only variants returning True are preserved.

    Returns
    -------
    varcode.VariantCollection
        Filtered variant collection, with only the variants passing the filter
    """
    if filter_fn:
        return variant_collection.clone_with_new_elements([
            variant
            for variant in variant_collection
            if filter_fn(FilterableVariant(
                        variant=variant,
                        variant_collection=variant_collection,
                        patient=patient,
                        ), **kwargs)
        ])
    else:
        return variant_collection