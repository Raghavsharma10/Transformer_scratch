def load_files(self) -> selectiontools.Selections:
        """Read all network files of the current working directory, structure
        their contents in a |selectiontools.Selections| object, and return it.
        """
        devicetools.Node.clear_all()
        devicetools.Element.clear_all()
        selections = selectiontools.Selections()
        for (filename, path) in zip(self.filenames, self.filepaths):
            # Ensure both `Node` and `Element`start with a `fresh` memory.
            devicetools.Node.extract_new()
            devicetools.Element.extract_new()
            try:
                info = runpy.run_path(path)
            except BaseException:
                objecttools.augment_excmessage(
                    f'While trying to load the network file `{path}`')
            try:
                node: devicetools.Node = info['Node']
                element: devicetools.Element = info['Element']
                selections += selectiontools.Selection(
                    filename.split('.')[0],
                    node.extract_new(),
                    element.extract_new())
            except KeyError as exc:
                raise RuntimeError(
                    f'The class {exc.args[0]} cannot be loaded from the '
                    f'network file `{path}`.')

        selections += selectiontools.Selection(
            'complete',
            info['Node'].query_all(),
            info['Element'].query_all())
        return selections