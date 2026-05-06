def load_if_not_loaded(widget, filenames, verbose=False, delay=0.1, force=False, local=True, evaluator=None):
    """
    Load a javascript file to the Jupyter notebook context,
    unless it was already loaded.
    """
    if evaluator is None:
        evaluator = EVALUATOR  # default if not specified.
    for filename in filenames:
        loaded = False
        if force or not filename in LOADED_JAVASCRIPT:
            js_text = get_text_from_file_name(filename, local)
            if verbose:
                print("loading javascript file", filename, "with", evaluator)
            evaluator(widget, js_text)
            LOADED_JAVASCRIPT.add(filename)
            loaded = True
        else:
            if verbose:
                print ("not reloading javascript file", filename)
        if loaded and delay > 0:
            if verbose:
                print ("delaying to allow JS interpreter to sync.")
            time.sleep(delay)