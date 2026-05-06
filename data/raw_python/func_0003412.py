def matchfirst(self, event):
        '''
        Return first match for this event
        
        :param event: an input event
        '''
        # 1. matches(self.index[ind], event)
        # 2. matches(self.any, event)
        # 3. self.matches
        if self.depth < len(event.indices):
            ind = event.indices[self.depth]
            if ind in self.index:
                m = self.index[ind].matchfirst(event)
                if m is not None:
                    return m
            if hasattr(self, 'any'):
                m = self.any.matchfirst(event)
                if m is not None:
                    return m
        if self._use_dict:
            for o, m in self.matchers_dict:
                if m is None or m.judge(event):
                    return o
        else:
            for o, m in self.matchers_list:
                if m is None or m.judge(event):
                    return o