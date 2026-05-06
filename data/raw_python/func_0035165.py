def fromjson(cls, json_string: str) -> 'Event':
        """Create a new Event from a from a psycopg2-pgevent event JSON.

        Parameters
        ----------
        json_string: str
            Valid psycopg2-pgevent event JSON.

        Returns
        -------
        Event
            Event created from JSON deserialization.

        """
        obj = json.loads(json_string)
        return cls(
            UUID(obj['event_id']),
            obj['event_type'],
            obj['schema_name'],
            obj['table_name'],
            obj['row_id']
        )