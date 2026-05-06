def get_proxy(self, proxystr=''):
        """
        Get the proxy given the option passed on the command line.
        If an empty string is passed it looks at the HTTP_PROXY
        environment variable.
        """
        if not proxystr:
            proxystr = os.environ.get('HTTP_PROXY', '')
        if proxystr:
            if '@' in proxystr:
                user_password, server_port = proxystr.split('@', 1)
                if ':' in user_password:
                    user, password = user_password.split(':', 1)
                else:
                    user = user_password
                    prompt = 'Password for %s@%s: ' % (user, server_port)
                    password = urllib.quote(getpass.getpass(prompt))
                return '%s:%s@%s' % (user, password, server_port)
            else:
                return proxystr
        else:
            return None