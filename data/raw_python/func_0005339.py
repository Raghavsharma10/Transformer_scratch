def set_id_in_fkeys(cls, payload):
        """
        Looks for any keys in the payload that end with either _id or _ids, signaling a foreign
        key field. For each foreign key field, checks whether the value is using the name of the
        record or the actual primary ID of the record (which may include the model abbreviation, i.e.
        B-1). If the former case, the name is replaced with
        the record's primary ID.

        Args:
            payload: `dict`. The payload to POST or PATCH.

        Returns:
            `dict`. The payload.
        """
        for key in payload:
            val = payload[key]
            if not val:
               continue
            if key.endswith("_id"):
                model = getattr(THIS_MODULE, cls.FKEY_MAP[key])
                rec_id = model.replace_name_with_id(name=val)
                payload[key] = rec_id
            elif key.endswith("_ids"):
                model = getattr(THIS_MODULE, cls.FKEY_MAP[key])
                rec_ids = []
                for v in val:
                   rec_id = model.replace_name_with_id(name=v)
                   rec_ids.append(rec_id)
                payload[key] = rec_ids
        return payload