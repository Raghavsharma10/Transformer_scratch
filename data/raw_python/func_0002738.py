def fetch_plos_images(article_doi, output_dir, document):
    """
    Fetch the images for a PLoS article from the internet.

    PLoS images are known through the inspection of <graphic> and
    <inline-graphic> elements. The information in these tags are then parsed
    into appropriate URLs for downloading.
    """
    log.info('Processing images for {0}...'.format(article_doi))

    #A dict of URLs for PLoS subjournals
    journal_urls = {'pgen': 'http://www.plosgenetics.org/article/{0}',
                    'pcbi': 'http://www.ploscompbiol.org/article/{0}',
                    'ppat': 'http://www.plospathogens.org/article/{0}',
                    'pntd': 'http://www.plosntds.org/article/{0}',
                    'pmed': 'http://www.plosmedicine.org/article/{0}',
                    'pbio': 'http://www.plosbiology.org/article/{0}',
                    'pone': 'http://www.plosone.org/article/{0}',
                    'pctr': 'http://clinicaltrials.ploshubs.org/article/{0}'}

    #Identify subjournal name for base URL
    subjournal_name = article_doi.split('.')[1]
    base_url = journal_urls[subjournal_name]

    #Acquire <graphic> and <inline-graphic> xml elements
    graphics = document.document.getroot().findall('.//graphic')
    graphics += document.document.getroot().findall('.//inline-graphic')

    #Begin to download
    log.info('Downloading images, this may take some time...')
    for graphic in graphics:
        nsmap = document.document.getroot().nsmap
        xlink_href = graphic.attrib['{' + nsmap['xlink'] + '}' + 'href']

        #Equations are handled a bit differently than the others
        #Here we decide that an image name starting with "e" is an equation
        if xlink_href.split('.')[-1].startswith('e'):
            resource = 'fetchObject.action?uri=' + xlink_href + '&representation=PNG'
        else:
            resource = xlink_href + '/largerimage'
        full_url = base_url.format(resource)
        try:
            image = urllib.request.urlopen(full_url)
        except urllib.error.HTTPError as e:
            if e.code == 503:  # Server overload error
                time.sleep(1)  # Wait a second
                try:
                    image = urllib.request.urlopen(full_url)
                except:
                    return False  # Happened twice, give up
            else:
                log.error('urllib.error.HTTPError {0}'.format(e.code))
                return False
        else:
            img_name = xlink_href.split('.')[-1] + '.png'
            img_path = os.path.join(output_dir, img_name)
            with open(img_path, 'wb') as output:
                output.write(image.read())
            log.info('Downloaded image {0}'.format(img_name))
    log.info('Done downloading images')
    return True