def show_view(self):
        """
        Show :attr:`LoginForm` form.
        """
        self.current.output['login_process'] = True
        if self.current.is_auth:
            self._do_upgrade()
        else:
            self.current.output['forms'] = LoginForm(current=self.current).serialize()