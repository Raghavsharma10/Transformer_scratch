def RestoreTaskStoreFactory(store_class, chunk_size, restore_file, save_file):
    """
    Restores a task store from file.
    """
    intm_results = np.load(restore_file)
    intm = intm_results[intm_results.files[0]]
    idx = np.isnan(intm).flatten().nonzero()[0]
    partitions = math.ceil(len(idx) / float(chunk_size))
    task_store = store_class(partitions, idx.tolist(), save_file)
    task_store.num_tasks = len(idx)
    # Also set up matrices for saving results
    for f in intm_results.files:
        task_store.__dict__[f] = intm_results[f]
    return task_store