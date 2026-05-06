def search_nodenames(self, *substrings: str, name: str = 'nodenames') -> \
            'Selection':
        """Return a new selection containing all nodes of the current
        selection with a name containing at least one of the given substrings.

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, pub, _ = prepare_full_example_2()

        Pass the (sub)strings as positional arguments and, optionally, the
        name of the newly created |Selection| object as a keyword argument:

        >>> test = pub.selections.complete.copy('test')
        >>> from hydpy import prepare_model
        >>> test.search_nodenames('dill', 'lahn_1')
        Selection("nodenames",
                  nodes=("dill", "lahn_1"),
                  elements=())

        Wrong string specifications result in errors like the following:

        >>> test.search_nodenames(['dill', 'lahn_1'])
        Traceback (most recent call last):
        ...
        TypeError: While trying to determine the nodes of selection \
`test` with names containing at least one of the given substrings \
`['dill', 'lahn_1']`, the following error occurred: 'in <string>' \
requires string as left operand, not list

        Method |Selection.select_nodenames| restricts the current selection
        to the one determined with the the method |Selection.search_nodenames|:

        >>> test.select_nodenames('dill', 'lahn_1')
        Selection("test",
                  nodes=("dill", "lahn_1"),
                  elements=("land_dill", "land_lahn_1", "land_lahn_2",
                            "land_lahn_3", "stream_dill_lahn_2",
                            "stream_lahn_1_lahn_2", "stream_lahn_2_lahn_3"))

        On the contrary, the method |Selection.deselect_nodenames| restricts
        the current selection to all devices not determined by the method
        |Selection.search_nodenames|:

        >>> pub.selections.complete.deselect_nodenames('dill', 'lahn_1')
        Selection("complete",
                  nodes=("lahn_2", "lahn_3"),
                  elements=("land_dill", "land_lahn_1", "land_lahn_2",
                            "land_lahn_3", "stream_dill_lahn_2",
                            "stream_lahn_1_lahn_2", "stream_lahn_2_lahn_3"))
        """
        try:
            selection = Selection(name)
            for node in self.nodes:
                for substring in substrings:
                    if substring in node.name:
                        selection.nodes += node
                        break
            return selection
        except BaseException:
            values = objecttools.enumeration(substrings)
            objecttools.augment_excmessage(
                f'While trying to determine the nodes of selection '
                f'`{self.name}` with names containing at least one '
                f'of the given substrings `{values}`')