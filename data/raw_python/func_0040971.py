def similar(obj1, obj2):
    """Calculate similarity between two (Comparable) objects."""
    Comparable.log(obj1, obj2, '%')
    similarity = obj1.similarity(obj2)
    Comparable.log(obj1, obj2, '%', result=similarity)
    return similarity