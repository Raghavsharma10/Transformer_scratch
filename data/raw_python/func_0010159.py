def write_part_images(url, raw_html, html, filename):
    """Write image file(s) associated with HTML to disk, substituting filenames.

    Keywords arguments:
    url -- the URL from which the HTML has been extracted from (str)
    raw_html -- unparsed HTML file content (list)
    html -- parsed HTML file content (lxml.html.HtmlElement) (default: None)
    filename -- the PART.html filename (str)

    Return raw HTML with image names replaced with local image filenames.
    """
    save_dirname = '{0}_files'.format(os.path.splitext(filename)[0])
    if not os.path.exists(save_dirname):
        os.makedirs(save_dirname)
    images = html.xpath('//img/@src')
    internal_image_urls = [x for x in images if x.startswith('/')]

    headers = {'User-Agent': random.choice(USER_AGENTS)}
    for img_url in images:
        img_name = img_url.split('/')[-1]
        if "?" in img_name:
            img_name = img_name.split('?')[0]
        if not os.path.splitext(img_name)[1]:
            img_name = '{0}.jpeg'.format(img_name)

        try:
            full_img_name = os.path.join(save_dirname, img_name)
            with open(full_img_name, 'wb') as img:
                if img_url in internal_image_urls:
                    # Internal images need base url added
                    full_img_url = '{0}{1}'.format(url.rstrip('/'), img_url)
                else:
                    # External image
                    full_img_url = img_url
                img_content = requests.get(full_img_url, headers=headers,
                                           proxies=get_proxies()).content
                img.write(img_content)
                raw_html = raw_html.replace(escape(img_url), full_img_name)
        except (OSError, IOError):
            pass
        time.sleep(random.uniform(0, 0.5))  # Slight delay between downloads
    return raw_html