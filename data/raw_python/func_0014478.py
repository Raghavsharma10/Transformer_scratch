def connect(self):
        """
        Method automatically called by the run() method of the AgentProxyThread
        """
        if ('SSH_AUTH_SOCK' in os.environ) and (sys.platform != 'win32'):
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                retry_on_signal(lambda: conn.connect(os.environ['SSH_AUTH_SOCK']))
            except:
                # probably a dangling env var: the ssh agent is gone
                return
        elif sys.platform == 'win32':
            import win_pageant
            if win_pageant.can_talk_to_agent():
                conn = win_pageant.PageantConnection()
            else:
                return
        else:
            # no agent support
            return
        self._conn = conn