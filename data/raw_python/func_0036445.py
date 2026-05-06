def generate_pagination(total_page_num, current_page_num):
    """
    >>> PAGE_SIZE = 10
    >>> generate_pagination(total_page_num=9, current_page_num=1)
    {'start': 1, 'end': 9, 'current': 1}
    >>> generate_pagination(total_page_num=20, current_page_num=12)
    {'start': 8, 'end': 17, 'current': 12}
    >>> generate_pagination(total_page_num=20, current_page_num=4)
    {'start': 1, 'end': 10, 'current': 4}
    >>> generate_pagination(total_page_num=16, current_page_num=14)
    {'start': 7, 'end': 16, 'current': 14}
    """
    pagination = {'start': 1, 'end': PAGE_SIZE, 'current': current_page_num}

    if total_page_num <= PAGE_SIZE:
        pagination['end'] = total_page_num
    else:
        # base on front four and back five
        pagination['start'] = current_page_num - 4
        pagination['end'] = current_page_num + 5

        if pagination['start'] < 1:
            pagination['start'] = 1
            pagination['end'] = PAGE_SIZE

        if pagination['end'] > total_page_num:
            pagination['end'] = total_page_num
            pagination['start'] = total_page_num - 9

    return pagination