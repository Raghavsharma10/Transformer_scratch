def _format_obj(self, item=None):
        """ Determines the type of the object and maps it to the correct
            formatter
        """
        # Order here matters, odd behavior with tuples
        if item is None:
            return getattr(self, 'number')(item)
        elif isinstance(item, self.str_):
            #: String
            return item + " "
        elif isinstance(item, bytes):
            #: Bytes
            return getattr(self, 'bytes')(item)
        elif isinstance(item, self.numeric_):
            #: Float, int, etc.
            return getattr(self, 'number')(item)
        elif isinstance(item, self.dict_):
            #: Dict
            return getattr(self, 'dict')(item)
        elif isinstance(item, self.list_):
            #: List
            return getattr(self, 'list')(item)
        elif isinstance(item, tuple):
            #: Tuple
            return getattr(self, 'tuple')(item)
        elif isinstance(item, types.GeneratorType):
            #: Generator
            return getattr(self, 'generator')(item)
        elif isinstance(item, self.set_):
            #: Set
            return getattr(self, 'set')(item)
        elif isinstance(item, deque):
            #: Deque
            return getattr(self, 'deque')(item)
        elif isinstance(item, Sequence):
            #: Sequence
            return getattr(self, 'sequence')(item)
        #: Any other object
        return getattr(self, 'object')(item)