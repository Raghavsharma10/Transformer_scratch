def login():
    " View function which handles an authentication request. "
    form = LoginForm(request.form)
    # make sure data are valid, but doesn't validate password is right
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        # we use werzeug to validate user's password
        if user and user.check_password(form.password.data):
            users.login(user)
            flash(_('Welcome %(user)s', user=user.username))
            return redirect(url_for('users.profile'))
        flash(_('Wrong email or password'), 'error-message')
    return redirect(request.referrer or url_for(users._login_manager.login_view))