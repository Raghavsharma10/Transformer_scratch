def getUserData(self,
                    hostgroup,
                    domain,
                    defaultPwd='',
                    defaultSshKey='',
                    proxyHostname='',
                    tplFolder='metadata/templates/'):
        """ Function getUserData
        Generate a userdata script for metadata server from Foreman API

        @param domain: the domain item linked to this host
        @param hostgroup: the hostgroup item linked to this host
        @param defaultPwd: the default password if no password is specified
                           in the host>hostgroup>domain params
        @param defaultSshKey: the default ssh key if no password is specified
                              in the host>hostgroup>domain params
        @param proxyHostname: hostname of the smartproxy
        @param tplFolder: the templates folder
        @return RETURN: the user data
        """
        if 'user-data' in self.keys():
            return self['user-data']
        else:
            self.hostgroup = hostgroup
            self.domain = domain
            if proxyHostname == '':
                proxyHostname = 'foreman.' + domain['name']
            password = self.getParamFromEnv('password', defaultPwd)
            sshauthkeys = self.getParamFromEnv('global_sshkey', defaultSshKey)
            with open(tplFolder+'puppet.conf', 'r') as puppet_file:
                p = MyTemplate(puppet_file.read())
                content = p.substitute(foremanHostname=proxyHostname)
                enc_puppet_file = base64.b64encode(bytes(content, 'utf-8'))
            with open(tplFolder+'cloud-init.tpl', 'r') as content_file:
                s = MyTemplate(content_file.read())
                if sshauthkeys:
                    sshauthkeys = ' - '+sshauthkeys
                self.userdata = s.substitute(
                    password=password,
                    fqdn=self['name'],
                    sshauthkeys=sshauthkeys,
                    foremanurlbuilt="http://{}/unattended/built"
                                    .format(proxyHostname),
                    puppet_conf_content=enc_puppet_file.decode('utf-8'))
                return self.userdata