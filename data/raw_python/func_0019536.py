def ec2_table(instances):
    """
    Print nice looking table of information from list of instances
    """
    t = prettytable.PrettyTable(['ID', 'State', 'Monitored', 'Image', 'Name', 'Type', 'SSH key', 'DNS'])
    t.align = 'l'
    for i in instances:
        name = i.tags.get('Name', '')
        t.add_row([i.id, i.state, i.monitored, i.image_id, name, i.instance_type, i.key_name, i.dns_name])
    return t