def load(self, name=None, *args, **kwargs):
        "Load the instance of the object from the stash."
        inst = self.stash.load(name)
        if inst is None:
            inst = self.instance(name, *args, **kwargs)
        logger.debug(f'loaded (conf mng) instance: {inst}')
        return inst