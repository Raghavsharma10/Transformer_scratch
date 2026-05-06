def grow_use_function(self, depth=0):
        "Select either function or terminal in grow method"
        if depth == 0:
            return False
        if depth == self._depth:
            return True
        return np.random.random() < 0.5