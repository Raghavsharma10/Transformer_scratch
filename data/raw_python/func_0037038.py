def parse(payload, candidate_classes):
        """ Parse a json response into an intent.

        :param payload: a JSON object representing an intent.
        :param candidate_classes: a list of classes representing various
                                  intents, each having their own `parse`
                                  method to attempt parsing the JSON object
                                  into the given intent class.
        :return: An object version of the intent if one of the candidate
                 classes managed to parse it, or None.
        """
        for cls in candidate_classes:
            intent = cls.parse(payload)
            if intent:
                return intent
        return None