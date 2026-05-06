def tojson(self) -> str:
        """Serialize an Event into JSON.

        Returns
        -------
        str
            JSON-serialized Event.

        """
        return json.dumps({
            'event_id': str(self.id),
            'event_type': self.type,
            'schema_name': self.schema_name,
            'table_name': self.table_name,
            'row_id': self.row_id
        })