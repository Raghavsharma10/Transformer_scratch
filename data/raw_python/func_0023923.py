def get_form(self, form_class=None):
        '''
        Set form groups to the groups specified in the view if defined
        '''
        formobj = super(GenModify, self).get_form(form_class)

        # Set requested group to this form
        selfgroups = getattr(self, "form_groups", None)
        if selfgroups:
            if type(selfgroups) == list:
                formobj.__groups__ = lambda: selfgroups
            else:
                formobj.__groups__ = selfgroups
        else:
            selfgroups = getattr(self, "__groups__", None)
            if selfgroups:
                formobj.__groups__ = selfgroups

        # Return the new updated form
        return formobj