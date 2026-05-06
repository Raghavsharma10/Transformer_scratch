def save(self):
        """ Creates a new user and account. Returns the newly created user. """
        username, email, password, first_name, last_name = (self.cleaned_data['username'],
                                     self.cleaned_data['email'],
                                     self.cleaned_data['password1'],
                                     self.cleaned_data['first_name'],
                                     self.cleaned_data['last_name'],)

        new_user = get_user_model()(username=username,
                                 email=email,
                                 first_name=first_name,
                                 last_name=last_name)
        new_user.set_password(password)
        new_user.save()
        return new_user