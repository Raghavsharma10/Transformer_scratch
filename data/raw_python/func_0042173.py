def zip(keys, values):
    """
    CONVERT LIST OF KEY/VALUE PAIRS TO Data
    PLEASE `import dot`, AND CALL `dot.zip()`
    """
    output = Data()
    for i, k in enumerate(keys):
        if i >= len(values):
            break
        output[k] = values[i]
    return output