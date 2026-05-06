def set_state(self, preset=None, **kwargs):
        """Set OpenGL rendering state, optionally using a preset
    
        Parameters
        ----------
        preset : str | None
            Can be one of ('opaque', 'translucent', 'additive') to use
            use reasonable defaults for these typical use cases.
        **kwargs : keyword arguments
            Other supplied keyword arguments will override any preset defaults.
            Options to be enabled or disabled should be supplied as booleans
            (e.g., ``'depth_test=True'``, ``cull_face=False``), non-boolean
            entries will be passed as arguments to ``set_*`` functions (e.g.,
            ``blend_func=('src_alpha', 'one')`` will call ``set_blend_func``).
    
        Notes
        -----
        This serves three purposes:
    
        1. Set GL state using reasonable presets.
        2. Wrapping glEnable/glDisable functionality.
        3. Convienence wrapping of other ``gloo.set_*`` functions.
    
        For example, one could do the following:
    
            >>> from vispy import gloo
            >>> gloo.set_state('translucent', depth_test=False, clear_color=(1, 1, 1, 1))  # noqa, doctest:+SKIP
    
        This would take the preset defaults for 'translucent', turn
        depth testing off (which would normally be on for that preset),
        and additionally set the glClearColor parameter to be white.
    
        Another example to showcase glEnable/glDisable wrapping:
    
            >>> gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=False)  # noqa, doctest:+SKIP
    
        This would be equivalent to calling
    
            >>> from vispy.gloo import gl
            >>> gl.glDisable(gl.GL_BLEND)
            >>> gl.glEnable(gl.GL_DEPTH_TEST)
            >>> gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
    
        Or here's another example:
    
            >>> gloo.set_state(clear_color=(0, 0, 0, 1), blend=True, blend_func=('src_alpha', 'one'))  # noqa, doctest:+SKIP
    
        Thus arbitrary GL state components can be set directly using
        ``set_state``. Note that individual functions are exposed e.g.,
        as ``set_clear_color``, with some more informative docstrings
        about those particular functions.
        """
        kwargs = deepcopy(kwargs)
        
        # Load preset, if supplied
        if preset is not None:
            _check_valid('preset', preset, tuple(list(_gl_presets.keys())))
            for key, val in _gl_presets[preset].items():
                # only overwrite user input with preset if user's input is None
                if key not in kwargs:
                    kwargs[key] = val
    
        # cull_face is an exception because GL_CULL_FACE, glCullFace both exist
        if 'cull_face' in kwargs:
            cull_face = kwargs.pop('cull_face')
            if isinstance(cull_face, bool):
                funcname = 'glEnable' if cull_face else 'glDisable'
                self.glir.command('FUNC', funcname, 'cull_face')
            else:
                self.glir.command('FUNC', 'glEnable', 'cull_face')
                self.set_cull_face(*_to_args(cull_face))
        
        # Iterate over kwargs
        for key, val in kwargs.items():
            if key in _setters:
                # Setter
                args = _to_args(val)
                # these actually need tuples
                if key in ('blend_color', 'clear_color') and \
                        not isinstance(args[0], string_types):
                    args = [args]
                getattr(self, 'set_' + key)(*args)
            else:
                # Enable / disable
                funcname = 'glEnable' if val else 'glDisable'
                self.glir.command('FUNC', funcname, key)