def search_upstream(self, device: devicetools.Device,
                        name: str = 'upstream') -> 'Selection':
        """Return the network upstream of the given starting point, including
        the starting point itself.

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, pub, _ = prepare_full_example_2()

        You can pass both |Node| and |Element| objects and, optionally,
        the name of the newly created |Selection| object:

        >>> test = pub.selections.complete.copy('test')
        >>> test.search_upstream(hp.nodes.lahn_2)
        Selection("upstream",
                  nodes=("dill", "lahn_1", "lahn_2"),
                  elements=("land_dill", "land_lahn_1", "land_lahn_2",
                            "stream_dill_lahn_2", "stream_lahn_1_lahn_2"))
        >>> test.search_upstream(
        ...     hp.elements.stream_lahn_1_lahn_2, 'UPSTREAM')
        Selection("UPSTREAM",
                  nodes="lahn_1",
                  elements=("land_lahn_1", "stream_lahn_1_lahn_2"))

        Wrong device specifications result in errors like the following:

        >>> test.search_upstream(1)
        Traceback (most recent call last):
        ...
        TypeError: While trying to determine the upstream network of \
selection `test`, the following error occurred: Either a `Node` or \
an `Element` object is required as the "outlet device", but the given \
`device` value is of type `int`.

        >>> pub.selections.headwaters.search_upstream(hp.nodes.lahn_3)
        Traceback (most recent call last):
        ...
        KeyError: "While trying to determine the upstream network of \
selection `headwaters`, the following error occurred: 'No node named \
`lahn_3` available.'"

        Method |Selection.select_upstream| restricts the current selection
        to the one determined with the method |Selection.search_upstream|:

        >>> test.select_upstream(hp.nodes.lahn_2)
        Selection("test",
                  nodes=("dill", "lahn_1", "lahn_2"),
                  elements=("land_dill", "land_lahn_1", "land_lahn_2",
                            "stream_dill_lahn_2", "stream_lahn_1_lahn_2"))

        On the contrary, the method |Selection.deselect_upstream| restricts
        the current selection to all devices not determined by method
        |Selection.search_upstream|:

        >>> complete = pub.selections.complete.deselect_upstream(
        ...     hp.nodes.lahn_2)
        >>> complete
        Selection("complete",
                  nodes="lahn_3",
                  elements=("land_lahn_3", "stream_lahn_2_lahn_3"))

        If necessary, include the "outlet device" manually afterwards:

        >>> complete.nodes += hp.nodes.lahn_2
        >>> complete
        Selection("complete",
                  nodes=("lahn_2", "lahn_3"),
                  elements=("land_lahn_3", "stream_lahn_2_lahn_3"))
        """
        try:
            selection = Selection(name)
            if isinstance(device, devicetools.Node):
                node = self.nodes[device.name]
                return self.__get_nextnode(node, selection)
            if isinstance(device, devicetools.Element):
                element = self.elements[device.name]
                return self.__get_nextelement(element, selection)
            raise TypeError(
                f'Either a `Node` or an `Element` object is required '
                f'as the "outlet device", but the given `device` value '
                f'is of type `{objecttools.classname(device)}`.')
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to determine the upstream network of '
                f'selection `{self.name}`')