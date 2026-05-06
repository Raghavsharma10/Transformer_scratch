def csv_to_list(csv_file):
    """
    Open and transform a CSV file into a matrix (list of lists).

    .. code:: python

        reusables.csv_to_list("example.csv")
        # [['Name', 'Location'],
        #  ['Chris', 'South Pole'],
        #  ['Harry', 'Depth of Winter'],
        #  ['Bob', 'Skull']]

    :param csv_file: Path to CSV file as str
    :return: list
    """
    with open(csv_file, 'r' if PY3 else 'rb') as f:
        return list(csv.reader(f))