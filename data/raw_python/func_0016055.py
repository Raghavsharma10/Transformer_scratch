def protocols():
    """Return a list of values from the IANA Service Name and Transport
    Protocol Port Number Registry, or an empty list if the IANA website is
    unreachable.
    Store it as a function attribute so that we only build the list once.
    """
    if not hasattr(protocols, 'protlist'):
        plist = []
        try:
            data = requests.get('http://www.iana.org/assignments/service-names'
                                '-port-numbers/service-names-port-numbers.csv')
        except requests.exceptions.RequestException:
            return []

        for line in data.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.count(',') > 0:
                    vals = line.split(',')
                    if vals[0]:
                        plist.append(vals[0])
                    if len(vals) > 2 and vals[2] and vals[2] not in plist:
                        plist.append(vals[2])

        plist.append('ipv4')
        plist.append('ipv6')
        plist.append('ssl')
        plist.append('tls')
        plist.append('dns')
        protocols.protlist = plist
    return protocols.protlist