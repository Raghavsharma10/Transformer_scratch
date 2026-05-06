def typed_encode(self, r):
        """
        :param record:  expecting id and value properties
        :return:  dict with id and json properties
        """
        try:
            value = r.get('value')
            if "json" in r:
                value = json2value(r["json"])
            elif is_data(value) or value != None:
                pass
            else:
                from mo_logs import Log
                raise Log.error("Expecting every record given to have \"value\" or \"json\" property")

            _buffer = UnicodeBuilder(1024)
            net_new_properties = []
            path = []
            if is_data(value):
                given_id = self.get_id(value)
                value['_id'] = None
                version = self.get_version(value)
            else:
                given_id = None
                version = None

            if given_id:
                record_id = r.get('id')
                if record_id and record_id != given_id:
                    from mo_logs import Log

                    raise Log.error(
                        "expecting {{property}} of record ({{record_id|quote}}) to match one given ({{given|quote}})",
                        property=self.id_info,
                        record_id=record_id,
                        given=given_id
                    )
            else:
                record_id = r.get('id')
                if record_id:
                    given_id = record_id
                else:
                    given_id = random_id()

            typed_encode(value, self.schema, path, net_new_properties, _buffer)
            json = _buffer.build()

            return given_id, version, json
        except Exception as e:
            # THE PRETTY JSON WILL PROVIDE MORE DETAIL ABOUT THE SERIALIZATION CONCERNS
            from mo_logs import Log

            Log.error("Serialization of JSON problems", cause=e)