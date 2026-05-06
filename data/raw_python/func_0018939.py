def search_modeltypes(self, *models: ModelTypesArg,
                          name: str = 'modeltypes') -> 'Selection':
        """Return a |Selection| object containing only the elements
        currently handling models of the given types.

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, pub, _ = prepare_full_example_2()

        You can pass both |Model| objects and names and, as a keyword
        argument, the name of the newly created |Selection| object:

        >>> test = pub.selections.complete.copy('test')
        >>> from hydpy import prepare_model
        >>> hland_v1 = prepare_model('hland_v1')

        >>> test.search_modeltypes(hland_v1)
        Selection("modeltypes",
                  nodes=(),
                  elements=("land_dill", "land_lahn_1", "land_lahn_2",
                            "land_lahn_3"))
        >>> test.search_modeltypes(
        ...     hland_v1, 'hstream_v1', 'lland_v1', name='MODELTYPES')
        Selection("MODELTYPES",
                  nodes=(),
                  elements=("land_dill", "land_lahn_1", "land_lahn_2",
                            "land_lahn_3", "stream_dill_lahn_2",
                            "stream_lahn_1_lahn_2", "stream_lahn_2_lahn_3"))

        Wrong model specifications result in errors like the following:

        >>> test.search_modeltypes('wrong')
        Traceback (most recent call last):
        ...
        ModuleNotFoundError: While trying to determine the elements of \
selection `test` handling the model defined by the argument(s) `wrong` \
of type(s) `str`, the following error occurred: \
No module named 'hydpy.models.wrong'

        Method |Selection.select_modeltypes| restricts the current selection to
        the one determined with the method the |Selection.search_modeltypes|:

        >>> test.select_modeltypes(hland_v1)
        Selection("test",
                  nodes=(),
                  elements=("land_dill", "land_lahn_1", "land_lahn_2",
                            "land_lahn_3"))

        On the contrary, the method |Selection.deselect_upstream| restricts
        the current selection to all devices not determined by method the
        |Selection.search_upstream|:

        >>> pub.selections.complete.deselect_modeltypes(hland_v1)
        Selection("complete",
                  nodes=(),
                  elements=("stream_dill_lahn_2", "stream_lahn_1_lahn_2",
                            "stream_lahn_2_lahn_3"))
        """
        try:
            typelist = []
            for model in models:
                if not isinstance(model, modeltools.Model):
                    model = importtools.prepare_model(model)
                typelist.append(type(model))
            typetuple = tuple(typelist)
            selection = Selection(name)
            for element in self.elements:
                if isinstance(element.model, typetuple):
                    selection.elements += element
            return selection
        except BaseException:
            values = objecttools.enumeration(models)
            classes = objecttools.enumeration(
                objecttools.classname(model) for model in models)
            objecttools.augment_excmessage(
                f'While trying to determine the elements of selection '
                f'`{self.name}` handling the model defined by the '
                f'argument(s) `{values}` of type(s) `{classes}`')