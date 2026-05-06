def select_distinct_field(col, field_or_fields, filters=None):
    """Select distinct value or combination of values of
    single or multiple fields.

    :params fields: str or list of str.
    :return data: list of list.

    **中文文档**

    选择多列中出现过的所有可能的排列组合。
    """
    fields = _preprocess_field_or_fields(field_or_fields)

    if filters is None:
        filters = dict()

    if len(fields) == 1:
        key = fields[0]
        data = list(col.find(filters).distinct(key))
        return data
    else:
        pipeline = [
            {
                "$match": filters
            },
            {
                "$group": {
                    "_id": {key: "$" + key for key in fields},
                },
            },
        ]
        data = list()
        for doc in col.aggregate(pipeline):
            # doc = {"_id": {"a": 0, "b": 0}} ...
            data.append([doc["_id"][key] for key in fields])
        return data