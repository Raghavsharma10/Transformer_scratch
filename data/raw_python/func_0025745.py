def triggerLogicToStr(self):
        """ Print all the trigger logic to a string and return it. """
        try:
            import json
        except ImportError:
            return "Cannot dump triggers/dependencies/executes (need json)"
        retval = "TRIGGERS:\n"+json.dumps(self._allTriggers, indent=3)
        retval += "\nDEPENDENCIES:\n"+json.dumps(self._allDepdcs, indent=3)
        retval += "\nTO EXECUTE:\n"+json.dumps(self._allExecutes, indent=3)
        retval += "\n"
        return retval