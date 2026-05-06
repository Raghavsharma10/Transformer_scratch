def add_substitution(self, short, medium, long, module):
        """Add the given substitutions both as a `short2long` and a
        `medium2long` mapping.

        Assume `variable1` is defined in the hydpy module `module1` and the
        short and medium descriptions are `var1` and `mod1.var1`:

        >>> import types
        >>> module1 = types.ModuleType('hydpy.module1')
        >>> from hydpy.core.autodoctools import Substituter
        >>> substituter = Substituter()
        >>> substituter.add_substitution(
        ...     'var1', 'mod1.var1', 'module1.variable1', module1)
        >>> print(substituter.get_commands())
        .. var1 replace:: module1.variable1
        .. mod1.var1 replace:: module1.variable1

        Adding `variable2` of `module2` has no effect on the predefined
        substitutions:

        >>> module2 = types.ModuleType('hydpy.module2')
        >>> substituter.add_substitution(
        ...     'var2', 'mod2.var2', 'module2.variable2', module2)
        >>> print(substituter.get_commands())
        .. var1 replace:: module1.variable1
        .. var2 replace:: module2.variable2
        .. mod1.var1 replace:: module1.variable1
        .. mod2.var2 replace:: module2.variable2

        But when adding `variable1` of `module2`, the `short2long` mapping
        of `variable1` would become inconclusive, which is why the new
        one (related to `module2`) is not stored and the old one (related
        to `module1`) is removed:

        >>> substituter.add_substitution(
        ...     'var1', 'mod2.var1', 'module2.variable1', module2)
        >>> print(substituter.get_commands())
        .. var2 replace:: module2.variable2
        .. mod1.var1 replace:: module1.variable1
        .. mod2.var1 replace:: module2.variable1
        .. mod2.var2 replace:: module2.variable2

        Adding `variable2` of `module2` accidentally again, does not
        result in any undesired side-effects:

        >>> substituter.add_substitution(
        ...     'var2', 'mod2.var2', 'module2.variable2', module2)
        >>> print(substituter.get_commands())
        .. var2 replace:: module2.variable2
        .. mod1.var1 replace:: module1.variable1
        .. mod2.var1 replace:: module2.variable1
        .. mod2.var2 replace:: module2.variable2

        In order to reduce the risk of name conflicts, only the
        `medium2long` mapping is supported for modules not part of the
        *HydPy* package:

        >>> module3 = types.ModuleType('module3')
        >>> substituter.add_substitution(
        ...     'var3', 'mod3.var3', 'module3.variable3', module3)
        >>> print(substituter.get_commands())
        .. var2 replace:: module2.variable2
        .. mod1.var1 replace:: module1.variable1
        .. mod2.var1 replace:: module2.variable1
        .. mod2.var2 replace:: module2.variable2
        .. mod3.var3 replace:: module3.variable3

        The only exception to this rule is |builtins|, for which only
        the `short2long` mapping is supported (note also, that the
        module name `builtins` is removed from string `long`):

        >>> import builtins
        >>> substituter.add_substitution(
        ...     'str', 'blt.str', ':func:`~builtins.str`', builtins)
        >>> print(substituter.get_commands())
        .. str replace:: :func:`str`
        .. var2 replace:: module2.variable2
        .. mod1.var1 replace:: module1.variable1
        .. mod2.var1 replace:: module2.variable1
        .. mod2.var2 replace:: module2.variable2
        .. mod3.var3 replace:: module3.variable3
        """
        name = module.__name__
        if 'builtin' in name:
            self._short2long[short] = long.split('~')[0] + long.split('.')[-1]
        else:
            if ('hydpy' in name) and (short not in self._blacklist):
                if short in self._short2long:
                    if self._short2long[short] != long:
                        self._blacklist.add(short)
                        del self._short2long[short]
                else:
                    self._short2long[short] = long
            self._medium2long[medium] = long