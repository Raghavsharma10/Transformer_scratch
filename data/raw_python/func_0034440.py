def info(self, section=None):
        """The INFO command returns information and statistics about the server
        in a format that is simple to parse by computers and easy to read by
        humans.

        The optional parameter can be used to select a specific section of
        information:

            - server: General information about the Redis server
            - clients: Client connections section
            - memory: Memory consumption related information
            - persistence: RDB and AOF related information
            - stats: General statistics
            - replication: Master/slave replication information
            - cpu: CPU consumption statistics
            - commandstats: Redis command statistics
            - cluster: Redis Cluster section
            - keyspace: Database related statistics

        It can also take the following values:

            - all: Return all sections
            - default: Return only the default set of sections

        When no parameter is provided, the default option is assumed.

        :param str section: Optional
        :return: dict

        """
        cmd = [b'INFO']
        if section:
            cmd.append(section)
        return self._execute(cmd, format_callback=common.format_info_response)