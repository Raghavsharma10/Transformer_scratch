def update_or_expire_session(self):
        """
        Deletes session if keepalive request expired
        otherwise updates the keepalive timestamp value
        """
        if not hasattr(self, 'key'):
            return
        now = time.time()
        timestamp = float(self.get() or 0) or now
        sess_id = self.sess_id or UserSessionID(self.user_id).get()
        if sess_id and now - timestamp > self.SESSION_EXPIRE_TIME:
            Session(sess_id).delete()
            return False
        else:
            self.set(now)
            return True