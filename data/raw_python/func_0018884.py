def save_files(self, selections) -> None:
        """Save the |Selection| objects contained in the given |Selections|
        instance to separate network files."""
        try:
            currentpath = self.currentpath
            selections = selectiontools.Selections(selections)
            for selection in selections:
                if selection.name == 'complete':
                    continue
                path = os.path.join(currentpath, selection.name+'.py')
                selection.save_networkfile(filepath=path)
        except BaseException:
            objecttools.augment_excmessage(
                'While trying to save selections `%s` into network files'
                % selections)