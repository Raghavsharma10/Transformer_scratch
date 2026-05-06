async def _create_remote_user(self, **payload):
        """
            This method creates a service record in the remote user service
            with the given email.
            Args:
                uid (str): the user identifier to create
            Returns:
                (dict): a summary of the user that was created
        """
        # the action for reading user entries
        read_action = get_crud_action(method='create', model='user')

        # see if there is a matching user
        user_data = await self.event_broker.ask(
            action_type=read_action,
            payload=payload
        )
        # treat the reply like a json object
        return json.loads(user_data)