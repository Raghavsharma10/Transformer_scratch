def insertAnnouncement(self, announcement):
        """
        Adds an announcement to the registry for later analysis.
        """
        url = announcement.get('url', None)
        try:
            peers.Peer(url)
        except:
            raise exceptions.BadUrlException(url)
        try:
            # TODO get more details about the user agent
            models.Announcement.create(
                url=announcement.get('url'),
                attributes=json.dumps(announcement.get('attributes', {})),
                remote_addr=announcement.get('remote_addr', None),
                user_agent=announcement.get('user_agent', None))
        except Exception as e:
            raise exceptions.RepoManagerException(e)