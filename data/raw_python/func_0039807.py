def moderators(self, sr, limit=None):
        """GETs moderators of subreddit ``sr``.  Returns :class:`things.ListBlob` object.
        
        **NOTE**: The :class:`things.Account` objects in the returned ListBlob *only* have ``id`` and ``name`` set.  This is because that's all reddit returns.  If you need full info on each moderator, you must individually GET them using :meth:`user` or :meth:`things.Account.about`.
        
        URL: ``http://www.reddit.com/r/<sr>/about/moderators/``
        
        :param sr: name of subreddit
        """
        userlist = self._limit_get('r', sr, 'about', 'moderators', limit=limit)
        return _process_userlist(userlist)