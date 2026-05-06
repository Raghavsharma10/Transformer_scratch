def safeReadJSON(url, logger, max_check=6, waittime=30):
    '''Return JSON object from URL'''
    counter = 0
    # try, try and try again ....
    while counter < max_check:
        try:
            with contextlib.closing(urllib.request.urlopen(url)) as f:
                res = json.loads(f.read().decode('utf8'))
            return res
        except Exception as errmsg:
            logger.info('----- GNR error [{0}] : retrying ----'.format(errmsg))
        counter += 1
        time.sleep(waittime)
    logger.error('----- Returning nothing : GNR server may be down -----')
    return None