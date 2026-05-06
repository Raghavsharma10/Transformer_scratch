def get(self, names, country_id=None, language_id=None, retheader=False):
        """
        Look up gender for a list of names.
        Can optionally refine search with locale info.
        May make multiple requests if there are more names than
        can be retrieved in one call.

        :param names: List of names.
        :type names: Iterable[str]
        :param country_id: Optional ISO 3166-1 alpha-2 country code.
        :type country_id: Optional[str]
        :param language_id: Optional ISO 639-1 language code.
        :type language_id: Optional[str]
        :param retheader: Optional
        :type retheader: Optional[boolean]
        :return:
        If retheader is False:
            List of dicts containing 'name', 'gender',
                     'probability', 'count' keys. If 'gender' is None,
                     'probability' and 'count' will be omitted.
        else:
            A dict containing 'data' and 'headers' keys.
            Data is the same as when retheader is False.
            Headers are the response header
            (a requests.structures.CaseInsensitiveDict).
            If multiple requests were made,
            the header will be from the last one.
        :rtype: Union[dict, Sequence[dict]]
        :raises GenderizeException: if API server returns HTTP error code.
        """
        responses = [
            self._get_chunk(name_chunk, country_id, language_id)
            for name_chunk
            in _chunked(names, Genderize.BATCH_SIZE)
        ]
        data = list(chain.from_iterable(
            response.data for response in responses
        ))
        if retheader:
            return {
                "data": data,
                "headers": responses[-1].headers,
            }
        else:
            return data