async def object_resolver(self, object_name, fields, obey_auth=False, current_user=None, **filters):
        """
            This function resolves a given object in the remote backend services
        """

        try:
            # check if an object with that name has been registered
            registered = [model for model in self._external_service_data['models'] \
                                if model['name']==object_name][0]
        # if there is no connection data yet
        except AttributeError:
            raise ValueError("No objects are registered with this schema yet.")
        # if we dont recognize the model that was requested
        except IndexError:
            raise ValueError("Cannot query for object {} on this service.".format(object_name))

        # the valid fields for this object
        valid_fields = [field['name'] for field in registered['fields']]

        # figure out if any invalid fields were requested
        invalid_fields = [field for field in fields if field not in valid_fields]
        try:
            # make sure we never treat pk as invalid
            invalid_fields.remove('pk')
        # if they weren't asking for pk as a field
        except ValueError:
            pass

        # if there were
        if invalid_fields:
            # yell loudly
            raise ValueError("Cannot query for fields {!r} on {}".format(
                invalid_fields, registered['name']
            ))

        # make sure we include the id in the request
        fields.append('pk')

        # the query for model records
        query = query_for_model(fields, **filters)

        # the action type for the question
        action_type = get_crud_action('read', object_name)

        # query the appropriate stream for the information
        response = await self.event_broker.ask(
            action_type=action_type,
            payload=query
        )

        # treat the reply like a json object
        response_data = json.loads(response)

        # if something went wrong
        if 'errors' in response_data and response_data['errors']:
            # return an empty response
            raise ValueError(','.join(response_data['errors']))

        # grab the valid list of matches
        result = response_data['data'][root_query()]

        # grab the auth handler for the object
        auth_criteria = self.auth_criteria.get(object_name)

        # if we care about auth requirements and there is one for this object
        if obey_auth and auth_criteria:

            # build a second list of authorized entries
            authorized_results = []

            # for each query result
            for query_result in result:
                # create a graph entity for the model
                graph_entity = GraphEntity(self, model_type=object_name, id=query_result['pk'])
                # if the auth handler passes
                if await auth_criteria(model=graph_entity, user_id=current_user):
                    # add the result to the final list
                    authorized_results.append(query_result)

            # overwrite the query result
            result = authorized_results

        # apply the auth handler to the result
        return result