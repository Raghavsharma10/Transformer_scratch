def table_add(tab, data, col):
    """
    Function to parse dictionary list **data** and add the data to table **tab** for column **col**

    Parameters
    ----------
    tab: Table class
      Table to store values
    data: list
      Dictionary list from the SQL query
    col: str
      Column name (ie, dictionary key) for the column to add

    """

    x = []
    for i in range(len(data)):

        # If the particular key is not present, use a place-holder value (used for photometry tables)
        if col not in data[i]:
            temp = ''
        else:
            temp = data[i][col]

        # Fix up None elements
        if temp is None: temp = ''

        x.append(temp)

    print('Adding column {}'.format(col))
    tab.add_column(Column(x, name=col))