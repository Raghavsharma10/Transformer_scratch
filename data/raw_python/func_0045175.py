def get_programme_title(pid):
    """Take BBC programme ID as string; returns programme title as string."""
    print("Extracting title and station...")
    main_page_etree = open_listing_page(pid)
    try:
        title, = main_page_etree.xpath('//title/text()')
    except ValueError:
        title = ''
    return title.strip()