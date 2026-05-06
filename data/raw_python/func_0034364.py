def clean_title(title):
    """
    Clean title -> remove dates, remove duplicated spaces and strip title.

    Args:
        title (str): Title.

    Returns:
        str: Clean title without dates, duplicated, trailing and leading spaces.

    """
    date_pattern = re.compile(r'\W*'
                              r'\d{1,2}'
                              r'[/\-.]'
                              r'\d{1,2}'
                              r'[/\-.]'
                              r'(?=\d*)(?:.{4}|.{2})'
                              r'\W*')
    title = date_pattern.sub(' ', title)
    title = re.sub(r'\s{2,}', ' ', title)
    title = title.strip()
    return title