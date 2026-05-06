async def async_get_program_guide(channel, no_cache=False, refresh_interval=4):
    '''
    Get the program data for a channel
    '''
    chan = await async_determine_channel(channel)
    now = datetime.datetime.now()
    max_cache_age = datetime.timedelta(hours=refresh_interval)
    if not no_cache and 'guide' in _CACHE and _CACHE.get('guide').get(chan):
        cache = _CACHE.get('guide').get(chan)
        cache_age = cache.get('last_updated')
        if now - cache_age < max_cache_age:
            _LOGGER.debug('Found program guide in cache.')
            return cache.get('data')
        else:
            _LOGGER.debug('Found outdated program guide in cache. Update it.')
            _CACHE['guide'].pop(chan)
    chans = await async_get_channels()
    url = chans.get('data', {}).get(chan)
    if not url:
        _LOGGER.error('Could not determine URL for %s', chan)
        return
    soup = await _async_request_soup(url)
    programs = []
    for prg_item in soup.find_all('div', {'class': 'program-infos'}):
        try:
            prog_info = prg_item.find('a', {'class': 'prog_name'})
            prog_name = prog_info.text.strip()
            prog_url = prog_info.get('href')
            if not prog_url:
                _LOGGER.warning('Failed to retrive the detail URL for program %s. '
                                'The summary will be empty', prog_name)
            prog_type = prg_item.find('span', {'class': 'prog_type'}).text.strip()
            prog_times = prg_item.find('div', {'class': 'prog_progress'})
            prog_start = datetime.datetime.fromtimestamp(
                int(prog_times.get('data-start')))
            prog_end = datetime.datetime.fromtimestamp(
                int(prog_times.get('data-end')))
            img = prg_item.find_previous_sibling().find(
                'img', {'class': 'prime_broadcast_image'})
            prog_img = img.get('data-src') if img else None
            programs.append(
                {'name': prog_name, 'type': prog_type, 'img': prog_img,
                 'url': prog_url, 'summary': None, 'start_time': prog_start,
                 'end_time': prog_end})
        except Exception as exc:
            _LOGGER.error('Exception occured while fetching the program '
                          'guide for channel %s: %s', chan, exc)
            import traceback
            traceback.print_exc()
    # Set the program summaries asynchronously
    tasks = [async_set_summary(prog) for prog in programs]
    programs = await asyncio.gather(*tasks)
    if programs:
        if 'guide' not in _CACHE:
            _CACHE['guide'] = {}
        _CACHE['guide'][chan] = {'last_updated': now, 'data': programs}
    return programs