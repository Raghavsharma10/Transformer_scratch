def _sort_handlers(cls, signals, handlers, configs):
        """Sort class defined handlers to give precedence to those declared at
        lower level. ``config`` can contain two keys ``begin`` or ``end`` that
        will further reposition the handler at the two extremes.
        """
        def macro_precedence_sorter(flags, hname):
            """The default is to sort 'bottom_up', with lower level getting
            executed first, but sometimes you need them reversed."""
            data = configs[hname]
            topdown_sort = SignalOptions.SORT_TOPDOWN in flags
            if topdown_sort:
                level = levels_count - 1 - data['level']
            else:
                level = data['level']
            if 'begin' in data:
                return (-1, level, hname)
            elif 'end' in data:
                return (1, level, hname)
            else:
                return (0, level, hname)

        levels_count = len(handlers.maps)
        per_signal = defaultdict(list)
        for level, m in enumerate(reversed(handlers.maps)):
            for hname, sig_name in m.items():
                sig_handlers = per_signal[sig_name]
                if hname not in sig_handlers:
                    configs[hname]['level'] = level
                    sig_handlers.append(hname)
        for sig_name, sig_handlers in per_signal.items():
            if sig_name in signals:  # it may be on a mixin
                flags = signals[sig_name].flags
                sig_handlers.sort(key=partial(macro_precedence_sorter,
                                              flags))
        return per_signal