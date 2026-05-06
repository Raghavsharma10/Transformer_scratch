def tab(tab_name, element_list=None, section_list=None):
    """
    Returns a dictionary representing a new tab to display elements.
    This can be thought of as a simple container for displaying multiple
    types of information.

    Args:
        tab_name: The title to display
        element_list: The list of elements to display. If a single element is
                      given it will be wrapped in a list.
        section_list: A list of sections to display.

    Returns:
        A dictionary with metadata specifying that it is to be rendered
        as a page containing multiple elements and/or tab.
    """
    _tab = {
            'Type': 'Tab',
            'Title': tab_name,
            }

    if element_list is not None:
        if isinstance(element_list, list):
            _tab['Elements'] = element_list
        else:
            _tab['Elements'] = [element_list]
    if section_list is not None:
        if isinstance(section_list, list):
            _tab['Sections'] = section_list
        else:
            if 'Elements' not in section_list:
                _tab['Elements'] = element_list
            else:
                _tab['Elements'].append(element_list)
    return _tab