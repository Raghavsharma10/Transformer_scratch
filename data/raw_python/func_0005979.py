def get_login_form_component(self):
        """Initializes and returns the login form component

        @rtype: LoginForm
        @return: Initialized component
        """
        self.dw.wait_until(
            lambda: self.dw.is_present(LoginForm.locators.form),
            failure_message='login form was never present so could not get the model '
                            'upload form component'
        )

        self.login_form = LoginForm(
            parent_page=self,
            element=self.dw.find(LoginForm.locators.form),
        )
        return self.login_form