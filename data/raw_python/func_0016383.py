def param_set(self, param_name, _value=None, **kwargs):
        """
        Setting parameter. if _value, we inject it directly. if not, we use all extra kwargs
        :param topic_name: name of the topic
        :param _value: optional value
        :param kwargs: each extra kwarg will be put in the value if structure matches
        :return:
        """
        #changing unicode to string ( testing stability of multiprocess debugging )
        if isinstance(param_name, unicode):
            param_name = unicodedata.normalize('NFKD', param_name).encode('ascii', 'ignore')

        _value = _value or {}

        if kwargs:
            res = self.param_svc.call(args=(param_name, kwargs,))
        elif _value is not None:
            res = self.param_svc.call(args=(param_name, _value,))
        else:   # if _msg_content is None the request is invalid.
                # just return something to mean False.
            res = 'WRONG SET'

        return res is None