def ec2_image_table(images):
    """
    Print nice looking table of information from images
    """
    t = prettytable.PrettyTable(['ID', 'State', 'Name', 'Owner', 'Root device', 'Is public', 'Description'])
    t.align = 'l'
    for i in images:
        t.add_row([i.id, i.state, i.name, i.ownerId, i.root_device_type, i.is_public, i.description])
    return t