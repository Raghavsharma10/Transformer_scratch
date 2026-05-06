def login_required(f, redirect_field_name=REDIRECT_FIELD_NAME, login_url=None):
    """
    Decorator that wraps django.contrib.auth.decorators.login_required, but supports extracting Shopify's authentication
    query parameters (`shop`, `timestamp`, `signature` and `hmac`) and passing them on to the login URL (instead of just
    wrapping them up and encoding them in to the `next` parameter).

    This is useful for ensuring that users are automatically logged on when they first access a page through the Shopify
    Admin, which passes these parameters with every page request to an embedded app.
    """

    @wraps(f)
    def wrapper(request, *args, **kwargs):
        if is_authenticated(request.user):
            return f(request, *args, **kwargs)

        # Extract the Shopify-specific authentication parameters from the current request.
        shopify_params = {
            k: request.GET[k]
            for k in ['shop', 'timestamp', 'signature', 'hmac']
            if k in request.GET
        }

        # Get the login URL.
        resolved_login_url = force_str(resolve_url(login_url or settings.LOGIN_URL))

        # Add the Shopify authentication parameters to the login URL.
        updated_login_url = add_query_parameters_to_url(resolved_login_url, shopify_params)

        django_login_required_decorator = django_login_required(redirect_field_name=redirect_field_name,
                                                                login_url=updated_login_url)
        return django_login_required_decorator(f)(request, *args, **kwargs)

    return wrapper