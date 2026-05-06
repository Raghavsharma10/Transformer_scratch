def enable_snmp(self, address, community_string):
        """Enables SNMP.

        uWSGI server embeds a tiny SNMP server that you can use to integrate
        your web apps with your monitoring infrastructure.

        * http://uwsgi.readthedocs.io/en/latest/SNMP.html

        .. note:: SNMP server is started in the master process after dropping the privileges.
            If you want it to listen on a privileged port, you can either use Capabilities on Linux,
            or use the ``as-root`` option to run the master process as root.

        :param str|unicode address: UDP address to bind to.

            Examples:

                * 192.168.1.1:2222

        :param str|unicode community_string: SNMP instance identifier to address it.

        """
        self._set('snmp', address)
        self._set('snmp-community', community_string)

        return self._section