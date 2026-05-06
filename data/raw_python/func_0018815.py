def add_module(self, module, cython=False):
        """Add the given module, its members, and their submembers.

        The first examples are based on the site-package |numpy|: which
        is passed to method |Substituter.add_module|:

        >>> from hydpy.core.autodoctools import Substituter
        >>> substituter = Substituter()
        >>> import numpy
        >>> substituter.add_module(numpy)

        Firstly, the module itself is added:

        >>> substituter.find('|numpy|')
        |numpy| :mod:`~numpy`

        Secondly, constants like |numpy.nan| are added:

        >>> substituter.find('|numpy.nan|')
        |numpy.nan| :const:`~numpy.nan`

        Thirdly, functions like |numpy.clip| are added:

        >>> substituter.find('|numpy.clip|')
        |numpy.clip| :func:`~numpy.clip`

        Fourthly, clases line |numpy.ndarray| are added:

        >>> substituter.find('|numpy.ndarray|')
        |numpy.ndarray| :class:`~numpy.ndarray`

        When adding Cython modules, the `cython` flag should be set |True|:

        >>> from hydpy.cythons import pointerutils
        >>> substituter.add_module(pointerutils, cython=True)
        >>> substituter.find('set_pointer')
        |PPDouble.set_pointer| \
:func:`~hydpy.cythons.autogen.pointerutils.PPDouble.set_pointer`
        |pointerutils.PPDouble.set_pointer| \
:func:`~hydpy.cythons.autogen.pointerutils.PPDouble.set_pointer`
        """
        name_module = module.__name__.split('.')[-1]
        short = ('|%s|'
                 % name_module)
        long = (':mod:`~%s`'
                % module.__name__)
        self._short2long[short] = long
        for (name_member, member) in vars(module).items():
            if self.consider_member(
                    name_member, member, module):
                role = self.get_role(member, cython)
                short = ('|%s|'
                         % name_member)
                medium = ('|%s.%s|'
                          % (name_module,
                             name_member))
                long = (':%s:`~%s.%s`'
                        % (role,
                           module.__name__,
                           name_member))
                self.add_substitution(short, medium, long, module)
                if inspect.isclass(member):
                    for name_submember, submember in vars(member).items():
                        if self.consider_member(
                                name_submember, submember, module, member):
                            role = self.get_role(submember, cython)
                            short = ('|%s.%s|'
                                     % (name_member,
                                        name_submember))
                            medium = ('|%s.%s.%s|'
                                      % (name_module,
                                         name_member,
                                         name_submember))
                            long = (':%s:`~%s.%s.%s`'
                                    % (role,
                                       module.__name__,
                                       name_member,
                                       name_submember))
                            self.add_substitution(short, medium, long, module)