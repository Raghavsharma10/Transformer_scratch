def insert(self, matcher, obj):
        '''
        Insert a new matcher
        
        :param matcher: an EventMatcher
        
        :param obj: object to return
        '''
        current = self.subtree(matcher, True)
        #current.matchers[(obj, matcher)] = None
        if current._use_dict:
            current.matchers_dict[(obj, matcher)] = None
        else:
            current.matchers_list.append((obj, matcher))
        return current