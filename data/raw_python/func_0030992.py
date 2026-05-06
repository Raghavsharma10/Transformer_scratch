def merge_layouts(layouts):
    ''' Utility function for merging multiple layouts.

    Args:
        layouts (list): A list of BIDSLayout instances to merge.
    Returns:
        A BIDSLayout containing merged files and entities.
    Notes:
        Layouts will be merged in the order of the elements in the list. I.e.,
        the first Layout will be updated with all values in the 2nd Layout,
        then the result will be updated with values from the 3rd Layout, etc.
        This means that order matters: in the event of entity or filename
        conflicts, later layouts will take precedence.
    '''
    layout = layouts[0].clone()

    for l in layouts[1:]:
        layout.files.update(l.files)
        layout.domains.update(l.domains)

        for k, v in l.entities.items():
            if k not in layout.entities:
                layout.entities[k] = v
            else:
                layout.entities[k].files.update(v.files)

    return layout