def count_variants_function_builder(function_name, filterable_variant_function=None):
    """
    Creates a function that counts variants that are filtered by the provided filterable_variant_function.
    The filterable_variant_function is a function that takes a filterable_variant and returns True or False.

    Users of this builder need not worry about applying e.g. the Cohort's default `filter_fn`. That will be applied as well.
    """
    @count_function
    def count(row, cohort, filter_fn, normalized_per_mb, **kwargs):
        def count_filter_fn(filterable_variant, **kwargs):
            assert filter_fn is not None, "filter_fn should never be None, but it is."
            return ((filterable_variant_function(filterable_variant) if filterable_variant_function is not None else True) and
                    filter_fn(filterable_variant, **kwargs))
        patient_id = row["patient_id"]
        return cohort.load_variants(
            patients=[cohort.patient_from_id(patient_id)],
            filter_fn=count_filter_fn,
            **kwargs)
    count.__name__ = function_name
    count.__doc__ = str("".join(inspect.getsourcelines(filterable_variant_function)[0])) if filterable_variant_function is not None else ""
    return count