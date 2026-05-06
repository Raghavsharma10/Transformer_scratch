def register():
    " Registration Form. "
    form = RegisterForm(request.form)
    if form.validate_on_submit():
        # create an user instance not yet stored in the database
        user = User(
            username=form.username.data,
            email=form.email.data,
            pw_hash=form.password.data)

        # Insert the record in our database and commit it
        db.session.add(user)
        db.session.commit()

        users.login(user)

        # flash will display a message to the user
        flash(_('Thanks for registering'))
        # redirect user to the 'home' method of the user module.
        return redirect(url_for('users.profile'))
    return render_template("users/register.html", form=form)