def get_db_references(cls, entry):
        """
        get list of `models.DbReference` from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.DbReference`
        """
        db_refs = []

        for db_ref in entry.iterfind("./dbReference"):

            db_ref_dict = {'identifier': db_ref.attrib['id'], 'type_': db_ref.attrib['type']}
            db_refs.append(models.DbReference(**db_ref_dict))

        return db_refs