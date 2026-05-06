def delete_files(self, selections) -> None:
        """Delete the network files corresponding to the given selections
        (e.g. a |list| of |str| objects or a |Selections| object)."""
        try:
            currentpath = self.currentpath
            for selection in selections:
                name = str(selection)
                if name == 'complete':
                    continue
                if not name.endswith('.py'):
                    name += '.py'
                path = os.path.join(currentpath, name)
                os.remove(path)
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to remove the network files of '
                f'selections `{selections}`')