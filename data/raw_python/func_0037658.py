def _extract_ajax_endpoints(self):
    
        ''' make a GET request to freeproxylists.com/elite.html '''
        url = '/'.join([DOC_ROOT, ELITE_PAGE])
        response = requests.get(url)
    
        ''' extract the raw HTML doc from the response '''
        raw_html = response.text
    
        ''' convert raw html into BeautifulSoup object '''
        soup = BeautifulSoup(raw_html, 'lxml')

        for url in soup.select('table tr td table tr td a'):
            if 'elite #' in url.text:
                yield '%s/load_elite_d%s' % (DOC_ROOT, url['href'].lstrip('elite/'))