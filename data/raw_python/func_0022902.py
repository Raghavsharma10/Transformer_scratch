def connect(self, callback, ref=False, position='first',
                before=None, after=None):
        """Connect this emitter to a new callback.

        Parameters
        ----------
        callback : function | tuple
            *callback* may be either a callable object or a tuple
            (object, attr_name) where object.attr_name will point to a
            callable object. Note that only a weak reference to ``object``
            will be kept.
        ref : bool | str
            Reference used to identify the callback in ``before``/``after``.
            If True, the callback ref will automatically determined (see
            Notes). If False, the callback cannot be referred to by a string.
            If str, the given string will be used. Note that if ``ref``
            is not unique in ``callback_refs``, an error will be thrown.
        position : str
            If ``'first'``, the first eligible position is used (that
            meets the before and after criteria), ``'last'`` will use
            the last position.
        before : str | callback | list of str or callback | None
            List of callbacks that the current callback should precede.
            Can be None if no before-criteria should be used.
        after : str | callback | list of str or callback | None
            List of callbacks that the current callback should follow.
            Can be None if no after-criteria should be used.

        Notes
        -----
        If ``ref=True``, the callback reference will be determined from:

            1. If ``callback`` is ``tuple``, the secend element in the tuple.
            2. The ``__name__`` attribute.
            3. The ``__class__.__name__`` attribute.

        The current list of callback refs can be obtained using
        ``event.callback_refs``. Callbacks can be referred to by either
        their string reference (if given), or by the actual callback that
        was attached (e.g., ``(canvas, 'swap_buffers')``).

        If the specified callback is already connected, then the request is
        ignored.

        If before is None and after is None (default), the new callback will
        be added to the beginning of the callback list. Thus the
        callback that is connected _last_ will be the _first_ to receive
        events from the emitter.
        """
        callbacks = self.callbacks
        callback_refs = self.callback_refs
        
        callback = self._normalize_cb(callback)
        
        if callback in callbacks:
            return
        
        # deal with the ref
        if isinstance(ref, bool):
            if ref:
                if isinstance(callback, tuple):
                    ref = callback[1]
                elif hasattr(callback, '__name__'):  # function
                    ref = callback.__name__
                else:  # Method, or other
                    ref = callback.__class__.__name__
            else:
                ref = None
        elif not isinstance(ref, string_types):
            raise TypeError('ref must be a bool or string')
        if ref is not None and ref in self._callback_refs:
            raise ValueError('ref "%s" is not unique' % ref)

        # positions
        if position not in ('first', 'last'):
            raise ValueError('position must be "first" or "last", not %s'
                             % position)

        # bounds
        bounds = list()  # upper & lower bnds (inclusive) of possible cb locs
        for ri, criteria in enumerate((before, after)):
            if criteria is None or criteria == []:
                bounds.append(len(callback_refs) if ri == 0 else 0)
            else:
                if not isinstance(criteria, list):
                    criteria = [criteria]
                for c in criteria:
                    count = sum([(c == cn or c == cc) for cn, cc
                                 in zip(callback_refs, callbacks)])
                    if count != 1:
                        raise ValueError('criteria "%s" is in the current '
                                         'callback list %s times:\n%s\n%s'
                                         % (criteria, count,
                                            callback_refs, callbacks))
                matches = [ci for ci, (cn, cc) in enumerate(zip(callback_refs,
                                                                callbacks))
                           if (cc in criteria or cn in criteria)]
                bounds.append(matches[0] if ri == 0 else (matches[-1] + 1))
        if bounds[0] < bounds[1]:  # i.e., "place before" < "place after"
            raise RuntimeError('cannot place callback before "%s" '
                               'and after "%s" for callbacks: %s'
                               % (before, after, callback_refs))
        idx = bounds[1] if position == 'first' else bounds[0]  # 'last'

        # actually add the callback
        self._callbacks.insert(idx, callback)
        self._callback_refs.insert(idx, ref)
        return callback