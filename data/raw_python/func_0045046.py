def add_target_with_env(self, environment, name=None):
        """Add an SCons target to this nest, with an SCons Environment

        The function decorated will be immediately called with three arguments:

        * ``environment``: A clone of the SCons environment, with variables
          populated for all values in the control dictionary, plus a variable
          ``OUTDIR``.
        * ``outdir``: The output directory
        * ``control``: The control dictionary

        Each result will be added to the respective control dictionary for
        later nests to access.

        Differs from :meth:`SConsWrap.add_target` only by the addition of the
        ``Environment`` clone.
        """
        def deco(func):
            def nestfunc(control):
                env = environment.Clone()
                for k, v in control.items():
                    if k in env:
                        logger.warn("Overwriting previously bound value %s=%s",
                                    k, env[k])
                    env[k] = v
                destdir = os.path.join(self.dest_dir, control['OUTDIR'])
                env['OUTDIR'] = destdir
                return [func(env, destdir, control)]
            key = name or func.__name__
            self.nest.add(key, nestfunc, create_dir=False)
            self._register_alias(key)
            return func
        return deco