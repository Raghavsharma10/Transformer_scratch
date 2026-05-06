async def async_get_channels(no_cache=False, refresh_interval=4):
    '''
    Get channel list and corresponding urls
    '''
    # Check cache
    now = datetime.datetime.now()
    max_cache_age = datetime.timedelta(hours=refresh_interval)
    if not no_cache and 'channels' in _CACHE:
        cache = _CACHE.get('channels')
        cache_age = cache.get('last_updated')
        if now - cache_age < max_cache_age:
            _LOGGER.debug('Found channel list in cache.')
            return cache
        else:
            _LOGGER.debug('Found outdated channel list in cache. Update it.')
            _CACHE.pop('channels')
    soup = await _async_request_soup(BASE_URL + '/plan.html')
    channels = {}
    for li_item in soup.find_all('li'):
        try:
            child = li_item.findChild()
            if not child or child.name != 'a':
                continue
            href = child.get('href')
            if not href or not href.startswith('/programme/chaine'):
                continue
            channels[child.get('title')] = BASE_URL + href
        except Exception as exc:
            _LOGGER.error('Exception occured while fetching the channel '
                          'list: %s', exc)
    if channels:
        _CACHE['channels'] = {'last_updated': now, 'data': channels}
        return _CACHE['channels']