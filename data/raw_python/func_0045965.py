def valid_for(whitelist):
    """ descriptor to check the genus type of an item, to see
    if the method is valid for that type
    From http://stackoverflow.com/questions/30809814/python-descriptors-with-arguments
    :param whitelist: list of OLX tag names, like 'chapter' or 'vertical'
    :return:
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args):
            valid_item = False
            try:
                if Id(self.my_osid_object_form._my_map['genusTypeId']).identifier in whitelist:
                    valid_item = True
            except AttributeError:
                if Id(self.my_osid_object._my_map['genusTypeId']).identifier in whitelist:
                    valid_item = True
            finally:
                if valid_item:
                    return func(self, *args)
                else:
                    raise IllegalState('Method not allowed for this object.')
        return wrapper
    return decorator