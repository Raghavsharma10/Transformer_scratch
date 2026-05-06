def handle_simple_sequencing(func):
    """decorator, deal with simple sequencing cases"""
    from .assessment import assessment_utilities

    def wrapper(*args, **kwargs):
        # re-order these things because have to delete the part after
        # removing it from the parent sequence map
        if 'create_assessment_part' in func.__name__:
            result = func(*args, **kwargs)
            assessment_utilities.update_parent_sequence_map(result)
        elif func.__name__ == 'delete_assessment_part':
            # args[1] is the assessment_part_id
            assessment_utilities.remove_from_parent_sequence_map(*args)
            result = func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper