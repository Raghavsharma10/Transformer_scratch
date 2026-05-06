def page(title, description, element_list=None, tab_list=None):
    """
    Returns a dictionary representing a new page to display elements.
    This can be thought of as a simple container for displaying multiple
    types of information. The ``section`` method can be used to create
    separate tabs.

    Args:
        title: The title to display
        description: A description of the section
        element_list: The list of elements to display. If a single element is
                      given it will be wrapped in a list.
        tab_list: A list of tabs to display.

    Returns:
        A dictionary with metadata specifying that it is to be rendered
        as a page containing multiple elements and/or tabs.
    """
    _page = {
             'Type': 'Page',
             'Title': title,
             'Description': description,
             'Data': {},
             }

    if element_list is not None:
        if isinstance(element_list, list):
            _page['Data']['Elements'] = element_list
        else:
            _page['Data']['Elements'] = [element_list]
    if tab_list is not None:
        if isinstance(tab_list, list):
            _page['Data']['Tabs'] = tab_list
        else:
            _page['Data']['Tabs'] = [tab_list]
    return _page