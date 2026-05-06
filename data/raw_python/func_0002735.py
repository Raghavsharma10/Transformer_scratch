def get_images(output_directory, explicit, input_path, config, parsed_article):
    """
    Main logic controller for the placement of images into the output directory

    Controlling logic for placement of the appropriate imager files into the
    EPUB directory. This function interacts with interface arguments as well as
    the local installation config.py file. These may change behavior of this
    function in terms of how it looks for images relative to the input, where it
    finds explicit images, whether it will attempt to download images, and
    whether successfully downloaded images will be stored in the cache.

    Parameters
    ----------
    output_directory : str
        The directory path where the EPUB is being constructed/output
    explicit : str
        A directory path to a user specified directory of images. Allows *
        wildcard expansion.
    input_path : str
        The absolute path to the input XML file.
    config : config module
        The imported configuration module
    parsed_article : openaccess_epub.article.Article object
        The Article instance for the article being converted to EPUB
    """
    #Split the DOI
    journal_doi, article_doi = parsed_article.doi.split('/')
    log.debug('journal-doi : {0}'.format(journal_doi))
    log.debug('article-doi : {0}'.format(article_doi))

    #Get the rootname for wildcard expansion
    rootname = utils.file_root_name(input_path)

    #Specify where to place the images in the output
    img_dir = os.path.join(output_directory,
                           'EPUB',
                           'images-{0}'.format(article_doi))
    log.info('Using {0} as image directory target'.format(img_dir))

    #Construct path to cache for article
    article_cache = os.path.join(config.image_cache, journal_doi, article_doi)

    #Use manual image directory, explicit images
    if explicit:
        success = explicit_images(explicit, img_dir, rootname, config)
        if success and config.use_image_cache:
            move_images_to_cache(img_dir, article_cache)
        #Explicit images prevents all other image methods
        return success

    #Input-Relative import, looks for any one of the listed options
    if config.use_input_relative_images:
        #Prevents other image methods only if successful
        if input_relative_images(input_path, img_dir, rootname, config):
            if config.use_image_cache:
                move_images_to_cache(img_dir, article_cache)
            return True

    #Use cache for article if it exists
    if config.use_image_cache:
        #Prevents other image methods only if successful
        if image_cache(article_cache, img_dir):
            return True

    #Download images from Internet
    if config.use_image_fetching:
        os.mkdir(img_dir)
        if journal_doi == '10.3389':
            fetch_frontiers_images(article_doi, img_dir)
            if config.use_image_cache:
                move_images_to_cache(img_dir, article_cache)
            return True
        elif journal_doi == '10.1371':
            success = fetch_plos_images(article_doi, img_dir, parsed_article)
            if success and config.use_image_cache:
                move_images_to_cache(img_dir, article_cache)
            return success
        else:
            log.error('Fetching images for this publisher is not supported!')
            return False
    return False