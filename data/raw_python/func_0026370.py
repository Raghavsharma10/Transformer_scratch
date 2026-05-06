def _find_references(model_name, references=None):
    """
    Iterate over model references for `model_name`
    and return a list of parent model specifications (including those of
    `model_name`, ordered from parent to child).
    """
    references = references or []
    references.append(model_name)

    ref = MODELS[model_name].get('reference')
    if ref is not None:
        _find_references(ref, references)

    parent_models = [m for m in references]
    parent_models.reverse()

    return parent_models