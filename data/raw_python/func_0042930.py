def wiki_download(url):
    '''
    scrape friendly: sleep 20 seconds between each request, cache each result.
    '''
    DOWNLOAD_TMPL = '../data/tv_and_movie_freqlist%s.html'
    freq_range = url[url.rindex('/')+1:]

    tmp_path = DOWNLOAD_TMPL % freq_range
    if os.path.exists(tmp_path):
        print('cached.......', url)
        with codecs.open(tmp_path, 'r', 'utf8') as f:
            return f.read(), True
    with codecs.open(tmp_path, 'w', 'utf8') as f:
        print('downloading...', url)
        req = urllib.request.Request(url, headers={
                'User-Agent': 'zxcvbn'
                })
        response = urllib.request.urlopen(req)
        result = response.read().decode('utf8')
        f.write(result)
        return result, False