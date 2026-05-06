def ipfix():
    """Return a list of values from the list of IANA IP Flow Information Export
    (IPFIX) Entities, or an empty list if the IANA website is unreachable.
    Store it as a function attribute so that we only build the list once.
    """
    if not hasattr(ipfix, 'ipflist'):
        ilist = []
        try:
            data = requests.get('http://www.iana.org/assignments/ipfix/ipfix-'
                                'information-elements.csv')
        except requests.exceptions.RequestException:
            return []

        for line in data.iter_lines():
            if line:
                line = line.decode("utf-8")
                if re.match(r'^\d+(,[a-zA-Z0-9]+){2},', line):
                    vals = line.split(',')
                    if vals[1]:
                        ilist.append(vals[1])

        ipfix.ipflist = ilist
    return ipfix.ipflist