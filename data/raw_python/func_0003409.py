def remove(self, matcher, obj):
        '''
        Remove the matcher
        
        :param matcher: an EventMatcher
        
        :param obj: the object to remove
        '''
        current = self.subtree(matcher, False)
        if current is None:
            return
        # Assume that this pair only appears once
        if current._use_dict:
            try:
                del current.matchers_dict[(obj, matcher)]
            except KeyError:
                pass
        else:
            if len(current.matchers_list) > 10:
                # Convert to dict
                current.matchers_dict = OrderedDict((v, None) for v in current.matchers_list
                                                    if v != (obj, matcher))
                current.matchers_list = None
                current._use_dict = True
            else:
                try:
                    current.matchers_list.remove((obj, matcher))
                except ValueError:
                    pass
        while ((not current.matchers_dict) if current._use_dict
               else (not current.matchers_list))\
                and not current.matchers_dict\
                and not hasattr(current,'any')\
                and not current.index and current.parent is not None:
            # remove self from parents
            ind = current.parentIndex
            if ind is None:
                del current.parent.any
            else:
                del current.parent.index[ind]
            p = current.parent
            current.parent = None
            current = p