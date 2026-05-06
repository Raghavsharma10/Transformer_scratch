def insertPeer(self, peer):
        """
        Accepts a peer datamodel object and adds it to the registry.
        """
        try:
            models.Peer.create(
                url=peer.getUrl(),
                attributes=json.dumps(peer.getAttributes()))
        except Exception as e:
            raise exceptions.RepoManagerException(e)