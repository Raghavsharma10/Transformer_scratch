def get_commands(self, source=None):
        """Return a string containing multiple `reStructuredText`
        replacements with the substitutions currently defined.

        Some examples based on the subpackage |optiontools|:

        >>> from hydpy.core.autodoctools import Substituter
        >>> substituter = Substituter()
        >>> from hydpy.core import optiontools
        >>> substituter.add_module(optiontools)

        When calling |Substituter.get_commands| with the `source`
        argument, the complete `short2long` and `medium2long` mappings
        are translated into replacement commands (only a few of them
        are shown):

        >>> print(substituter.get_commands())
        .. |Options.autocompile| replace:: \
:const:`~hydpy.core.optiontools.Options.autocompile`
        .. |Options.checkseries| replace:: \
:const:`~hydpy.core.optiontools.Options.checkseries`
        ...
        .. |optiontools.Options.warntrim| replace:: \
:const:`~hydpy.core.optiontools.Options.warntrim`
        .. |optiontools.Options| replace:: \
:class:`~hydpy.core.optiontools.Options`

        Through passing a string (usually the source code of a file
        to be documented), only the replacement commands relevant for
        this string are translated:

        >>> from hydpy.core import objecttools
        >>> import inspect
        >>> source = inspect.getsource(objecttools)
        >>> print(substituter.get_commands(source))
        .. |Options.reprdigits| replace:: \
:const:`~hydpy.core.optiontools.Options.reprdigits`
        """
        commands = []
        for key, value in self:
            if (source is None) or (key in source):
                commands.append('.. %s replace:: %s' % (key, value))
        return '\n'.join(commands)