def GET_getitemtypes(self) -> None:
        """Get the types of all current exchange items supposed to return
        the values of |Parameter| or |Sequence| objects or the time series
        of |IOSequence| objects."""
        for item in state.getitems:
            type_ = self._get_itemtype(item)
            for name, _ in item.yield_name2value():
                self._outputs[name] = type_