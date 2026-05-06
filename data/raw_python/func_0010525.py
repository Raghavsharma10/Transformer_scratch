def _logout(self):
        """Logout from current session(token)
        This functions doesn't return any boolean value, since it can 'fail' for anonymous logins
        """
        self.log.debug("Logging out from session ID: %s" % self._token)
        try:
            info = self._xmlrpc_server.LogOut(self._token)
            self.log.debug("Logout ended in %s with status: %s" %
                           (info['seconds'], info['status']))
        except ProtocolError as e:
            self.log.debug("error in HTTP/HTTPS transport layer")
            raise
        except Fault as e:
            self.log.debug("error in xml-rpc server")
            raise
        except:
            self.log.exception("Connection to the server failed/other error")
            raise
        finally:
            # force token reset
            self._token = None