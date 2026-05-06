def count_function(func):
    """
    Decorator for functions that return a collection (technically a dict of collections) that should be
    counted up. Also automatically falls back to the Cohort-default filter_fn and normalized_per_mb if
    not specified.
    """
    # Fall back to Cohort-level defaults.
    @use_defaults
    @wraps(func)
    def wrapper(row, cohort, filter_fn=None, normalized_per_mb=None, **kwargs):
        per_patient_data = func(row=row,
                                cohort=cohort,
                                filter_fn=filter_fn,
                                normalized_per_mb=normalized_per_mb,
                                **kwargs)
        patient_id = row["patient_id"]
        if patient_id in per_patient_data:
            count = len(per_patient_data[patient_id])
            if normalized_per_mb:
                count /= float(get_patient_to_mb(cohort)[patient_id])
            return count
        return np.nan
    return wrapper