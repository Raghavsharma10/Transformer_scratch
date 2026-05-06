def display_google_book(id, page=None, width=700, height=500, **kwargs):
    """Display an embedded version of a Google book.
    :param id: the id of the google book to display.
    :type id: string
    :param page: the start page for the book.
    :type id: string or int."""
    if isinstance(page, int):
        url = 'http://books.google.co.uk/books?id={id}&pg=PA{page}&output=embed'.format(id=id, page=page)
    else:
        url = 'http://books.google.co.uk/books?id={id}&pg={page}&output=embed'.format(id=id, page=page)
    display_iframe_url(url, width=width, height=height, **kwargs)