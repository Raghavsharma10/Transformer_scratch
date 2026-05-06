async def mutation_resolver(self, mutation_name, args, fields):
        """
            the default behavior for mutations is to look up the event,
            publish the correct event type with the args as the body,
            and return the fields contained in the result
        """

        try:
            # make sure we can identify the mutation
            mutation_summary = [mutation for mutation in \
                                            self._external_service_data['mutations'] \
                                            if mutation['name'] == mutation_name][0]
        # if we couldn't get the first entry in the list
        except KeyError as e:
            # make sure the error is reported
            raise ValueError("Could not execute mutation named: " + mutation_name)


        # the function to use for running the mutation depends on its schronicity
        # event_function = self.event_broker.ask \
        #                     if mutation_summary['isAsync'] else self.event_broker.send
        event_function = self.event_broker.ask

        # send the event and wait for a response
        value =  await event_function(
            action_type=mutation_summary['event'],
            payload=args
        )
        try:
            # return a dictionary with the values we asked for
            return json.loads(value)

        # if the result was not valid json
        except json.decoder.JSONDecodeError:
            # just throw the value
            raise RuntimeError(value)