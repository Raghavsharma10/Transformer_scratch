def _generate_state() -> str:
        """
        Generates a new random string to be used as OAuth state.
        :return: A randomly generated OAuth state.
        """
        state = str(uuid.uuid4()).replace('-', '')
        logger.debug("Generated OAuth state: %s" % state)
        return state