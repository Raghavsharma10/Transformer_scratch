def search_windows(
            self, winname=None, winclass=None, winclassname=None,
            pid=None, only_visible=False, screen=None, require=False,
            searchmask=0, desktop=None, limit=0, max_depth=-1):
        """
        Search for windows.

        :param winname:
            Regexp to be matched against window name
        :param winclass:
            Regexp to be matched against window class
        :param winclassname:
            Regexp to be matched against window class name
        :param pid:
            Only return windows from this PID
        :param only_visible:
            If True, only return visible windows
        :param screen:
            Search only windows on this screen
        :param require:
            If True, will match ALL conditions. Otherwise, windows matching
            ANY condition will be returned.
        :param searchmask:
            Search mask, for advanced usage. Leave this alone if you
            don't kwnow what you are doing.
        :param limit:
            Maximum number of windows to list. Zero means no limit.
        :param max_depth:
            Maximum depth to return. Defaults to -1, meaning "no limit".
        :return:
            A list of window ids matching query.
        """
        windowlist_ret = ctypes.pointer(window_t(0))
        nwindows_ret = ctypes.c_uint(0)

        search = xdo_search_t(searchmask=searchmask)

        if winname is not None:
            search.winname = winname
            search.searchmask |= SEARCH_NAME

        if winclass is not None:
            search.winclass = winclass
            search.searchmask |= SEARCH_CLASS

        if winclassname is not None:
            search.winclassname = winclassname
            search.searchmask |= SEARCH_CLASSNAME

        if pid is not None:
            search.pid = pid
            search.searchmask |= SEARCH_PID

        if only_visible:
            search.only_visible = True
            search.searchmask |= SEARCH_ONLYVISIBLE

        if screen is not None:
            search.screen = screen
            search.searchmask |= SEARCH_SCREEN

        if screen is not None:
            search.screen = desktop
            search.searchmask |= SEARCH_DESKTOP

        search.limit = limit
        search.max_depth = max_depth

        _libxdo.xdo_search_windows(
            self._xdo, search,
            ctypes.byref(windowlist_ret),
            ctypes.byref(nwindows_ret))

        return [windowlist_ret[i] for i in range(nwindows_ret.value)]