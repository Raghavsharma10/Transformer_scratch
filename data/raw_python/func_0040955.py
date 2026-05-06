def csv_row_cleaner(rows):
    """
    Clean row checking:
     - Not empty row.
     - >=1 element different in a row.
     - row allready in cleaned row result.


    """
    result = []

    for row in rows:

        # check not empty row
        check_empty = len(exclude_empty_values(row)) > 1

        # check more or eq than 1 unique element in row
        check_set = len(set(exclude_empty_values(row))) > 1
        # check row not into result cleaned rows.
        check_last_allready = (result and result[-1] == row)

        if check_empty and check_set and not check_last_allready:
            result.append(row)
    return result