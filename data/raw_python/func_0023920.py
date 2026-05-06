def get_context_json(self, context):
        '''
        Return a base answer for a json answer
        '''
        # Initialize answer
        answer = {}

        # Metadata builder
        answer['meta'] = self.__jcontext_metadata(context)

        # Filter builder
        answer['filter'] = self.__jcontext_filter(context)

        # Head builder
        answer['table'] = {}
        answer['table']['head'] = self.__jcontext_tablehead(context)
        answer['table']['body'] = None
        answer['table']['header'] = None
        answer['table']['summary'] = None

        # Return answer
        return answer