def addPeer(self):
        """
        Adds a new peer into this repo
        """
        self._openRepo()
        try:
            peer = peers.Peer(
                self._args.url, json.loads(self._args.attributes))
        except exceptions.BadUrlException:
            raise exceptions.RepoManagerException("The URL for the peer was "
                                                  "malformed.")
        except ValueError as e:
            raise exceptions.RepoManagerException(
                "The attributes message "
                "was malformed. {}".format(e))
        self._updateRepo(self._repo.insertPeer, peer)