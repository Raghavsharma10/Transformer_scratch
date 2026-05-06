def section(title, element_list):
    """
    Returns a dictionary representing a new section.  Sections
    contain a list of elements that are displayed separately from
    the global elements on the page.

    Args:
        title: The title of the section to be displayed
        element_list: The list of elements to display within the section

    Returns:
        A dictionary with metadata specifying that it is to be rendered as
        a section containing multiple elements
    """
    sect = {
            'Type': 'Section',
            'Title': title,
            }

    if isinstance(element_list, list):
        sect['Elements'] = element_list
    else:
        sect['Elements'] = [element_list]
    return sect