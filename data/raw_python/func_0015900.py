def set_vhost_permissions(self, vname, username, config, rd, wr):
        """
        Set permissions for a given username on a given vhost. Both
        must already exist.

        :param string vname: Name of the vhost to set perms on.
        :param string username: User to set permissions for.
        :param string config: Permission pattern for configuration operations
            for this user in this vhost.
        :param string rd: Permission pattern for read operations for this user
            in this vhost
        :param string wr: Permission pattern for write operations for this user
            in this vhost.

        Permission patterns are regex strings. If you're unfamiliar with this,
        you should definitely check out this section of the RabbitMQ docs:

        http://www.rabbitmq.com/admin-guide.html#access-control
        """
        vname = quote(vname, '')
        body = json.dumps({"configure": config, "read": rd, "write": wr})
        path = Client.urls['vhost_permissions'] % (vname, username)
        return self._call(path, 'PUT', body,
                                 headers=Client.json_headers)