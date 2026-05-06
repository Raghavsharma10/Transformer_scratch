def list_to_csv(my_list, csv_file):
    """
    Save a matrix (list of lists) to a file as a CSV

    .. code:: python

        my_list = [["Name", "Location"],
                   ["Chris", "South Pole"],
                   ["Harry", "Depth of Winter"],
                   ["Bob", "Skull"]]

        reusables.list_to_csv(my_list, "example.csv")

    example.csv

    .. code:: csv

        "Name","Location"
        "Chris","South Pole"
        "Harry","Depth of Winter"
        "Bob","Skull"

    :param my_list: list of lists to save to CSV
    :param csv_file: File to save data to
    """
    if PY3:
        csv_handler = open(csv_file, 'w', newline='')
    else:
        csv_handler = open(csv_file, 'wb')

    try:
        writer = csv.writer(csv_handler, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerows(my_list)
    finally:
        csv_handler.close()