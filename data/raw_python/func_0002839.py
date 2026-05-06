def initialize(self, maxsize, history=None):
        '''size specifies the maximum amount of history to keep'''
        super().__init__()

        self.maxsize = int(maxsize)
        self.history = deque(maxlen=self.maxsize)  # Preserves order history

        # If `items` are specified, then initialize with them
        if history is not None:
            for key, value in history:
                self.insert(key, value)