def eval_script(self, expr):
    """ Evaluates a piece of Javascript in the context of the current page and
    returns its value. """
    ret = self.conn.issue_command("Evaluate", expr)
    return json.loads("[%s]" % ret)[0]