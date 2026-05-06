def _dictify(self, doc):
        """Transforms the replies to a regular Python dict with
        strings and datetimes.

        Tested with BankID version 2.5 return data.

        :param doc: The response as interpreted by :py:mod:`zeep`.
        :returns: The response parsed to a dict.
        :rtype: dict

        """
        return {
            k: (self._dictify(doc[k]) if hasattr(doc[k], "_xsd_type") else doc[k])
            for k in doc
        }