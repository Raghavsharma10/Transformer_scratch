def _parse_spacy_kwargs(**kwargs):
    """Supported args include:

    Args:
        n_threads/num_threads: Number of threads to use. Uses num_cpus - 1 by default.
        batch_size: The number of texts to accumulate into a common working set before processing.
            (Default value: 1000)
    """
    n_threads = kwargs.get('n_threads') or kwargs.get('num_threads')
    batch_size = kwargs.get('batch_size')

    if n_threads is None or n_threads is -1:
        n_threads = cpu_count() - 1
    if batch_size is None or batch_size is -1:
        batch_size = 1000
    return n_threads, batch_size