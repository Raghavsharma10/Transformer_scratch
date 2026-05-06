def logout():
    " View function which handles a logout request. "
    users.logout()
    return redirect(request.referrer or url_for(users._login_manager.login_view))