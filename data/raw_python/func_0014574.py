def end_session(self, end_tcsession=True, sid=None):
        """End this test session.

        A session can be ended in three ways, depending on the value of the
        end_tcsession parameter:
            - end_tcsession=None:
                Stop using session locally, do not contact server.
            - end_tcsession=False:
                End client controller, but leave test session on server.
            - end_tcsession=True:
                End client controller and terminate test session (default).
            - end_tcsession='kill':
                Forcefully terminate test session.

        Specifying end_tcsession=False is useful to do before attaching an STC
        GUI or legacy automation script, so that there are not multiple
        controllers to interfere with each other.

        When the session is ended, it is no longer available.  Clients should
        export any result or log files, that they want to preserve, before the
        session is ended.

        Arguments
        end_tcsession -- How to end the session (see above)
        sid           -- ID of session to end.  None to use current session.

        Return:
        True if session ended, false if session was not started.

        """
        if not sid or sid == self._sid:
            if not self.started():
                return False

            sid = self._sid
            self._sid = None
            self._rest.del_header('X-STC-API-Session')

        if end_tcsession is None:
            if self._dbg_print:
                print('===> detached from session')
            return True

        try:
            if end_tcsession:
                if self._dbg_print:
                    print('===> deleting session:', sid)
                if end_tcsession == 'kill':
                    status, data = self._rest.delete_request(
                        'sessions', sid, 'kill')
                else:
                    status, data = self._rest.delete_request('sessions', sid)
                count = 0
                while 1:
                    time.sleep(5)
                    if self._dbg_print:
                        print('===> checking if session ended')
                    ses_list = self.sessions()
                    if not ses_list or sid not in ses_list:
                        break
                    count += 1
                    if count == 3:
                        raise RuntimeError("test session has not stopped")

                if self._dbg_print:
                    print('===> ok - deleted test session')
            else:
                # Ending client session is supported on version >= 2.1.5
                if self._get_api_version() < (2, 1, 5):
                    raise RuntimeError('option no available on server')

                status, data = self._rest.delete_request(
                    'sessions', sid, 'false')
                if self._dbg_print:
                    print('===> OK - detached REST API from test session')
        except resthttp.RestHttpError as e:
            raise RuntimeError('failed to end session: ' + str(e))

        return True