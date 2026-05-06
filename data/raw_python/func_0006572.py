def protected_view(view, info):
    """allows adding `protected=True` to a view_config`"""

    if info.options.get('protected'):
        def wrapper_view(context, request):
            response = _advice(request)
            if response is not None:
                return response
            else:
                return view(context, request)
        return wrapper_view
    return view