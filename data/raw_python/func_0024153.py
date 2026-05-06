def _do_upgrade(self):
        """ open websocket connection """
        self.current.output['cmd'] = 'upgrade'
        self.current.output['user_id'] = self.current.user_id
        self.terminate_existing_login()
        self.current.user.bind_private_channel(self.current.session.sess_id)
        user_sess = UserSessionID(self.current.user_id)
        user_sess.set(self.current.session.sess_id)
        self.current.user.is_online(True)
        # Clean up the locale from session to allow it to be re-read from the user preferences after login
        for k in translation.DEFAULT_PREFS.keys():
            self.current.session[k] = ''