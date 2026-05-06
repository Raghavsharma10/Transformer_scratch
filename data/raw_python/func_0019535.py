def elb_table(balancers):
    """
    Print nice looking table of information from list of load balancers
    """
    t = prettytable.PrettyTable(['Name', 'DNS', 'Ports', 'Zones', 'Created'])
    t.align = 'l'
    for b in balancers:
        ports = ['%s: %s -> %s' % (l[2], l[0], l[1]) for l in b.listeners]
        ports = '\n'.join(ports)
        zones = '\n'.join(b.availability_zones)
        t.add_row([b.name, b.dns_name, ports, zones, b.created_time])
    return t