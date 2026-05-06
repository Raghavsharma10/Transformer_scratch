def do_view(self):
        """
        Authenticate user with given credentials.
        Connects user's queue and exchange
        """
        self.current.output['login_process'] = True
        self.current.task_data['login_successful'] = False
        if self.current.is_auth:
            self._do_upgrade()
        else:
            try:
                auth_result = self.current.auth.authenticate(
                    self.current.input['username'],
                    self.current.input['password'])
                self.current.task_data['login_successful'] = auth_result
                if auth_result:
                    self._do_upgrade()
            except ObjectDoesNotExist:
                self.current.log.exception("Wrong username or another error occurred")
                pass
            except:
                raise
            if self.current.output.get('cmd') != 'upgrade':
                self.current.output['status_code'] = 403
            else:
                KeepAlive(self.current.user_id).reset()