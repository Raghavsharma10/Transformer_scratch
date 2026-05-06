def count_effects_function_builder(function_name, only_nonsynonymous, filterable_effect_function=None):
    """
    Create a function that counts effects that are filtered by the provided filterable_effect_function.
    The filterable_effect_function is a function that takes a filterable_effect and returns True or False.

    Users of this builder need not worry about applying e.g. the Cohort's default `filter_fn`. That will be applied as well.
    """
    @count_function
    def count(row, cohort, filter_fn, normalized_per_mb, **kwargs):
        def count_filter_fn(filterable_effect, **kwargs):
            assert filter_fn is not None, "filter_fn should never be None, but it is."
            return ((filterable_effect_function(filterable_effect) if filterable_effect_function is not None else True) and
                    filter_fn(filterable_effect, **kwargs))
        # This only loads one effect per variant.
        patient_id = row["patient_id"]
        return cohort.load_effects(
            only_nonsynonymous=only_nonsynonymous,
            patients=[cohort.patient_from_id(patient_id)],
            filter_fn=count_filter_fn,
            **kwargs)
    count.__name__ = function_name
    count.__doc__ = (("only_nonsynonymous=%s\n" % only_nonsynonymous) +
                     str("".join(inspect.getsourcelines(filterable_effect_function)[0])) if filterable_effect_function is not None else "")
    # Keep track of these to be able to query the returned function for these attributes
    count.only_nonsynonymous = only_nonsynonymous
    count.filterable_effect_function = filterable_effect_function
    return count