def majority_value(data, class_attr):
    """
    Creates a list of all values in the target attribute for each record
    in the data list object, and returns the value that appears in this list
    the most frequently.
    """
    if is_continuous(data[0][class_attr]):
        return CDist(seq=[record[class_attr] for record in data])
    else:
        return most_frequent([record[class_attr] for record in data])