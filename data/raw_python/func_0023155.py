def restore(self, state):
        """Restore the state of a mesh previously saved using save()

        Parameters
        ----------
        state : dict
            The previous state.
        """
        import pickle
        state = pickle.loads(state)
        for k in state:
            if isinstance(state[k], list):
                state[k] = np.array(state[k])
            setattr(self, k, state[k])