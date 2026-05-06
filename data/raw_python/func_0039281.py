def build_sort():
    '''Build sort query paramter from kwargs'''
    sorts = request.args.getlist('sort')
    sorts = [sorts] if isinstance(sorts, basestring) else sorts
    sorts = [s.split(' ') for s in sorts]
    return [{SORTS[s]: d} for s, d in sorts if s in SORTS]