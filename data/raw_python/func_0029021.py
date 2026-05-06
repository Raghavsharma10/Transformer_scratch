def get_confirmation_url(email, request, name="email_registration_confirm", **kwargs):
    """
    Returns the confirmation URL
    """
    return request.build_absolute_uri(
        reverse(name, kwargs={"code": get_confirmation_code(email, request, **kwargs)})
    )