def update_variable(self, variable, value) -> None:
        """Assign the given value(s) to the given target or base variable.

        If the assignment fails, |ChangeItem.update_variable| raises an
        error like the following:

        >>> from hydpy.core.examples import prepare_full_example_2
        >>> hp, pub, TestIO = prepare_full_example_2()
        >>> item = SetItem('alpha', 'hland_v1', 'control.alpha', 0)
        >>> item.collect_variables(pub.selections)
        >>> item.update_variables()    # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: When trying to update a target variable of SetItem `alpha` \
with the value(s) `None`, the following error occurred: While trying to set \
the value(s) of variable `alpha` of element `...`, the following error \
occurred: The given value `None` cannot be converted to type `float`.
        """
        try:
            variable(value)
        except BaseException:
            objecttools.augment_excmessage(
                f'When trying to update a target variable of '
                f'{objecttools.classname(self)} `{self.name}` '
                f'with the value(s) `{value}`')