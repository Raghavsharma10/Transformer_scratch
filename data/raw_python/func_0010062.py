def convert_weka_to_py_date_pattern(p):
    """
    Converts the date format pattern used by Weka to the date format pattern used by Python's datetime.strftime().
    """
    # https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
    # https://www.cs.waikato.ac.nz/ml/weka/arff.html
    p = p.replace('yyyy', r'%Y')
    p = p.replace('MM', r'%m')
    p = p.replace('dd', r'%d')
    p = p.replace('HH', r'%H')
    p = p.replace('mm', r'%M')
    p = p.replace('ss', r'%S')
    return p