def search_by(url=None, file=None):
    """
    TODO: support file
    """

    image_url = url
    # image_file = file

    """
    Search result page
    """

    result_url = GOOGLE_SEARCH_BY_ENDPOINT + image_url

    referer = 'http://www.google.com/imghp'
    result_html = fire_request(result_url, referer)

    result = SBIResult()
    result.result_page = result_url
    result.best_guess = extract_best_guess(result_html)

    soup = cook_soup(result_html)

    all_sizes_a_tag = soup.find('a', text='All sizes')

    # No other sizes of this image found
    if not all_sizes_a_tag:
        return result

    all_sizes_href = all_sizes_a_tag['href']
    all_sizes_url = urlparse.urljoin(GOOGLE_BASE_URL, all_sizes_href)

    result.all_sizes_page = all_sizes_url

    """
    All sizes page
    """

    all_sizes_html = fire_request(all_sizes_url, referer=all_sizes_url)

    soup = cook_soup(all_sizes_html)

    img_links =  soup.find_all('a', {'class': 'rg_l'})
    images = []
    for a in img_links:
        url = a['href']
        parse_result = urlparse.urlparse(url)

        querystring = parse_result.query
        querystring_dict = urlparse.parse_qs(querystring)

        image = {}
        image['url'] = querystring_dict['imgurl'][0]
        image['width'] = int(querystring_dict['w'][0])
        image['height'] = int(querystring_dict['h'][0])

        images.append(image)

    result.images = images

    return result