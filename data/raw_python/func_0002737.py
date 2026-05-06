def fetch_frontiers_images(doi, output_dir):
    """
    Fetch the images from Frontiers' website. This method may fail to properly
    locate all the images and should be avoided if the files can be accessed
    locally. Downloading the images to an appropriate directory in the cache,
    or to a directory specified by passed argument are the preferred means to
    access images.
    """
    log.info('Fetching Frontiers images')
    log.warning('This method may fail to locate all images.')

    def download_image(fetch, img_file):
        try:
            image = urllib.request.urlopen(fetch)
        except urllib.error.HTTPError as e:
            if e.code == 503:  # Server overloaded
                time.sleep(1)  # Wait one second
                try:
                    image = urllib.request.urlopen(fetch)
                except:
                    return None
            elif e.code == 500:
                print('urllib.error.HTTPError {0}'.format(e.code))
            return None
        else:
            with open(img_file, 'wb') as outimage:
                outimage.write(image.read())
        return True

    def check_equation_completion(images):
        """
        In some cases, equations images are not exposed in the fulltext (hidden
        behind a rasterized table). This attempts to look for gaps and fix them
        """
        log.info('Checking for complete equations')
        files = os.listdir(output_dir)
        inline_equations = []
        for e in files:
            if e[0] == 'i':
                inline_equations.append(e)
        missing = []
        highest = 0
        if inline_equations:
            inline_equations.sort()
            highest = int(inline_equations[-1][1:4])
            i = 1
            while i < highest:
                name = 'i{0}.gif'.format(str(i).zfill(3))
                if name not in inline_equations:
                    missing.append(name)
                i += 1
        get = images[0][:-8]
        for m in missing:
            loc = os.path.join(output_dir, m)
            download_image(get + m, loc)
            print('Downloaded image {0}'.format(loc))
        #It is possible that we need to go further than the highest
        highest += 1
        name = 'i{0}.gif'.format(str(highest).zfill(3))
        loc = os.path.join(output_dir, name)
        while download_image(get + name, loc):
            print('Downloaded image {0}'.format(loc))
            highest += 1
            name = 'i{0}.gif'.format(str(highest).zfill(3))

    print('Processing images for {0}...'.format(doi))
    #We use the DOI of the article to locate the page.
    doistr = 'http://dx.doi.org/{0}'.format(doi)
    logging.debug('Accessing DOI address-{0}'.format(doistr))
    page = urllib.request.urlopen(doistr)
    if page.geturl()[-8:] == 'abstract':
        full = page.geturl()[:-8] + 'full'
    elif page.geturl()[-4:] == 'full':
        full = page.geturl()
    print(full)
    page = urllib.request.urlopen(full)
    with open('temp', 'w') as temp:
        temp.write(page.read())
    images = []
    with open('temp', 'r') as temp:
        for l in temp.readlines():
            images += re.findall('<a href="(?P<href>http://\w{7}.\w{3}.\w{3}.rackcdn.com/\d{5}/f\w{4}-\d{2}-\d{5}-HTML/image_m/f\w{4}-\d{2}-\d{5}-\D{1,2}\d{3}.\D{3})', l)
            images += re.findall('<a href="(?P<href>http://\w{7}.\w{3}.\w{3}.rackcdn.com/\d{5}/f\w{4}-\d{2}-\d{5}-r2/image_m/f\w{4}-\d{2}-\d{5}-\D{1,2}\d{3}.\D{3})', l)
            images += re.findall('<img src="(?P<src>http://\w{7}.\w{3}.\w{3}.rackcdn.com/\d{5}/f\w{4}-\d{2}-\d{5}-HTML/image_n/f\w{4}-\d{2}-\d{5}-\D{1,2}\d{3}.\D{3})', l)
    os.remove('temp')
    for i in images:
        loc = os.path.join(output_dir, i.split('-')[-1])
        download_image(i, loc)
        print('Downloaded image {0}'.format(loc))
    if images:
        check_equation_completion(images)
    print("Done downloading images")