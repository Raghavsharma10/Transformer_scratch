def patch_view_model(view_cls, model_cls):
    """ Patches view_cls.Model with model_cls.

    :param view_cls: View class "Model" param of which should be
        patched
    :param model_cls: Model class which should be used to patch
        view_cls.Model
    """
    original_model = view_cls.Model
    view_cls.Model = model_cls

    try:
        yield
    finally:
        view_cls.Model = original_model