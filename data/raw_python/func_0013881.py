def to_disk_indices(value):
    """Convert following input to disk indices

    Sample input:

    ```
    Disks:
    Bus 0 Enclosure 0 Disk 9
    Bus 1 Enclosure 0 Disk 12
    Bus 1 Enclosure 0 Disk 9
    Bus 0 Enclosure 0 Disk 4
    Bus 0 Enclosure 0 Disk 7
    ```

    :param value: disk list
    :return: disk indices in list
    """
    ret = []
    p = re.compile(r'Bus\s+(\w+)\s+Enclosure\s+(\w+)\s+Disk\s+(\w+)')
    if value is not None:
        for line in value.split('\n'):
            line = line.strip()
            if len(line) == 0:
                continue
            matched = re.search(p, line)
            if matched is None or len(matched.groups()) < 3:
                continue
            else:
                ret.append('{}_{}_{}'.format(*matched.groups()))
    return ret