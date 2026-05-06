def get_context_for_help_msgs(self, context_dict):
        """ We override this method from HelpMsgMixIn to replace wrapped_func with its name """
        context_dict = copy(context_dict)
        context_dict['wrapped_func'] = get_callable_name(context_dict['wrapped_func'])
        return context_dict