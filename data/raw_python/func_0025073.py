def data_item_live(self, data_item):
        """ Return a context manager to put the data item in a 'live state'. """
        class LiveContextManager:
            def __init__(self, manager, object):
                self.__manager = manager
                self.__object = object
            def __enter__(self):
                self.__manager.begin_data_item_live(self.__object)
                return self
            def __exit__(self, type, value, traceback):
                self.__manager.end_data_item_live(self.__object)
        return LiveContextManager(self, data_item)