def authenticate(self, request, application, method):
        """Authenticate the AJAX request.

        By default any request to fetch a model is allowed for any user,
        including anonymous users. All other methods minimally require that
        the user is already logged in.

        Most likely you will want to lock down who can edit and delete various
        models. To do this, just override this method in your child class.
        """
        return self.authentication.is_authenticated(request, application, method)