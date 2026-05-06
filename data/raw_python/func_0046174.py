def display_login_info(sender, user, request, **kwargs):
    """
    Create a message, after login.

    Because this signal receiver will be called **after** auth.models.update_last_login(), the
    user.last_login information was updated before!

    As a work-a-round, we add **user.previous_login** in forms.SecureLoginForm.clean()
    """
    if not hasattr(user, "previous_login"):
        # e.g. normal django admin login page was used
        return
    message = render_to_string('secure_js_login/login_info.html', {"last_login":user.previous_login})
    messages.success(request, message)