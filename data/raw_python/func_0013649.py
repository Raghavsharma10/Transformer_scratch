def colon_separated_string_to_dict(string, separator=':'):
    '''
    Converts a string in the format:

        Name: Et3
        Switchport: Enabled
        Administrative Mode: trunk
        Operational Mode: trunk
        MAC Address Learning: enabled
        Access Mode VLAN: 3 (VLAN0003)
        Trunking Native Mode VLAN: 1 (default)
        Administrative Native VLAN tagging: disabled
        Administrative private VLAN mapping: ALL
        Trunking VLANs Enabled: 2-3,5-7,20-21,23,100-200
        Trunk Groups:

    into a dictionary

    '''
    dictionary = dict()
    for line in string.splitlines():
        line_data = line.split(separator)
        if len(line_data) > 1:
            dictionary[line_data[0].strip()] = ''.join(line_data[1:]).strip()
        elif len(line_data) == 1:
            dictionary[line_data[0].strip()] = None
        else:
            raise Exception('Something went wrong parsing the colo separated string {}'
                            .format(line))
    return dictionary