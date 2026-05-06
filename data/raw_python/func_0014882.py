def eval(self, expression, identify_erros=True):
    """ Evaluates a matlab expression synchronously.

    If identify_erros is true, and the last output line after evaluating the
    expressions begins with '???' an excpetion is thrown with the matlab error
    following the '???'.
    The return value of the function is the matlab output following the call.
    """
    #print expression
    self._check_open()
    ret = self.client.Execute(expression)
    #print ret
    if identify_erros and ret.rfind('???') != -1:
      begin = ret.rfind('???') + 4
      raise MatlabError(ret[begin:])
    return ret