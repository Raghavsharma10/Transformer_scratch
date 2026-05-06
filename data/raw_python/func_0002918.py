def progress(progress):
    """Convert given progress to a JSON object.

    Check that progress can be represented as float between 0 and 1 and
    return it in JSON of the form:

        {"proc.progress": progress}

    """
    if isinstance(progress, int) or isinstance(progress, float):
        progress = float(progress)
    else:
        try:
            progress = float(json.loads(progress))
        except (TypeError, ValueError):
            return warning("Progress must be a float.")

    if not 0 <= progress <= 1:
        return warning("Progress must be a float between 0 and 1.")

    return json.dumps({'proc.progress': progress})