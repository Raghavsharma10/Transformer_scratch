def replace_name_with_id(cls, name):
        """
        Used to replace a foreign key reference using a name with an ID. Works by searching the
        record in Pulsar and expects to find exactly one hit. First, will check if the foreign key
        reference is an integer value and if so, returns that as it is presumed to be the foreign key.

        Raises:
            `pulsarpy.elasticsearch_utils.MultipleHitsException`: Multiple hits were returned from the name search.
            `pulsarpy.models.RecordNotFound`: No results were produced from the name search.
        """
        try:
            int(name)
            return name #Already a presumed ID.
        except ValueError:
            pass
        #Not an int, so maybe a combination of MODEL_ABBR and Primary Key, i.e. B-8.
        if name.split("-")[0] in Meta._MODEL_ABBREVS:
            return int(name.split("-", 1)[1])
        try:
            result = cls.ES.get_record_by_name(cls.ES_INDEX_NAME, name)
            if result:
                return result["id"]
        except pulsarpy.elasticsearch_utils.MultipleHitsException as e:
            raise
        raise RecordNotFound("Name '{}' for model '{}' not found.".format(name, cls.__name__))