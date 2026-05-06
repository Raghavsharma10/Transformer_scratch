def _from_dict_dict(cls, dic):
        """Takes a dict {id : dict_attributes} """
        return cls({_convert_id(i): v for i, v in dic.items()})