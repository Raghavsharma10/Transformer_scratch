def _find_best_chat_server(servers, stats):
        """Find the best from servers by comparing with the stats

        :param servers: a list if server adresses, e.g. ['0.0.0.0:80']
        :type servers: :class:`list` of :class:`str`
        :param stats: list of server statuses
        :type stats: :class:`list` of :class:`chat.ChatServerStatus`
        :returns: the best server adress
        :rtype: :class:`str`
        :raises: None
        """
        best = servers[0]  # In case we sind no match with any status
        stats.sort()  # gets sorted for performance
        for stat in stats:
            for server in servers:
                if server == stat:
                    # found a chatserver that has the same address
                    # than one of the chatserverstats.
                    # since the stats are sorted for performance
                    # the first hit is the best, thus break
                    best = server
                    break
            if best:
                # already found one, so no need to check the other
                # statuses, which are worse
                break
        return best