def removePeer(self):
        """
        Removes a peer by URL from this repo
        """
        self._openRepo()

        def func():
            self._updateRepo(self._repo.removePeer, self._args.url)
        self._confirmDelete("Peer", self._args.url, func)