def get_info(self, full=False):
        " Return printable information about current site. "

        if full:
            context = self.as_dict()
            return "".join("{0:<25} = {1}\n".format(
                           key, context[key]) for key in sorted(context.iterkeys()))
        return "%s [%s]" % (self.get_name(), self.template)