def flushing_disabled(self):
        """Return a simplified one-word answer to the question, 'Has 
        automatic transaction log flushing been disabled on all indexes in 
        the cluster?'

        The answer will be one of "disabled" (yes, on all), "enabled" (no, 
        on all), "some" (yes, only on some), or "unknown".

        """
        states = self.get_index_translog_disable_flush().values()
        if not states:
            return "unknown"
        if all(s == True for s in states):
            return "disabled"
        if all(s == False for s in states):
            return "enabled"
        if any(s == False for s in states):
            return "some"
        return "unknown"