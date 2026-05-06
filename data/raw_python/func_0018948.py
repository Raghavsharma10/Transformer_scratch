def save_networkfile(self, filepath: Union[str, None] = None,
                         write_nodes: bool = True) -> None:
        """Save the selection as a network file.

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> _, pub, TestIO = prepare_full_example_2()

        In most cases, one should conveniently write network files via method
        |NetworkManager.save_files| of class |NetworkManager|.  However,
        using the method |Selection.save_networkfile| allows for additional
        configuration via the arguments `filepath` and `write_nodes`:

        >>> with TestIO():
        ...     pub.selections.headwaters.save_networkfile()
        ...     with open('headwaters.py') as networkfile:
        ...         print(networkfile.read())
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy import Node, Element
        <BLANKLINE>
        <BLANKLINE>
        Node("dill", variable="Q",
             keywords="gauge")
        <BLANKLINE>
        Node("lahn_1", variable="Q",
             keywords="gauge")
        <BLANKLINE>
        <BLANKLINE>
        Element("land_dill",
                outlets="dill",
                keywords="catchment")
        <BLANKLINE>
        Element("land_lahn_1",
                outlets="lahn_1",
                keywords="catchment")
        <BLANKLINE>

        >>> with TestIO():
        ...     pub.selections.headwaters.save_networkfile('test.py', False)
        ...     with open('test.py') as networkfile:
        ...         print(networkfile.read())
        # -*- coding: utf-8 -*-
        <BLANKLINE>
        from hydpy import Node, Element
        <BLANKLINE>
        <BLANKLINE>
        Element("land_dill",
                outlets="dill",
                keywords="catchment")
        <BLANKLINE>
        Element("land_lahn_1",
                outlets="lahn_1",
                keywords="catchment")
        <BLANKLINE>
        """
        if filepath is None:
            filepath = self.name + '.py'
        with open(filepath, 'w', encoding="utf-8") as file_:
            file_.write('# -*- coding: utf-8 -*-\n')
            file_.write('\nfrom hydpy import Node, Element\n\n')
            if write_nodes:
                for node in self.nodes:
                    file_.write('\n' + repr(node) + '\n')
                file_.write('\n')
            for element in self.elements:
                file_.write('\n' + repr(element) + '\n')