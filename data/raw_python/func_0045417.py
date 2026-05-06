def get_common_parts(r):
    """Gets citation parts which are common to all types of citation"""

    title = format_title(r.get('title'))
    author_list = format_author_list(r.get('author'))
    container = format_container(r.get('container-title'))
    date = format_date(r.get('issued'))
    doi = r.get('DOI')

    return Parts(type='Unknown', title=title, authors=author_list, container=container, date=date, extra='', doi=doi)