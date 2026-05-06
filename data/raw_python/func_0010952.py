def formatted_value(self):
        """ Returns a formatted value as a string"""
        # TODO: Cleanup all of this, it's just weird and unnatural maths
        val = self.value
        pval = val
        ftype = self.value_type

        if ftype == "percentage":
            pval = int(round(val * 100))

            if self.type == "negative":
                pval = 0 - (100 - pval)
            else:
                pval -= 100
        elif ftype == "additive_percentage":
            pval = int(round(val * 100))
        elif ftype == "inverted_percentage":
            pval = 100 - int(round(val * 100))

            # Can't remember what workaround this was, is it needed?
            if self.type == "negative":
                if self.value > 1:
                    pval = 0 - pval
        elif ftype == "additive" or ftype == "particle_index" or ftype == "account_id":
            if int(val) == val:
                pval = int(val)
        elif ftype == "date":
            d = time.gmtime(int(val))
            pval = time.strftime("%Y-%m-%d %H:%M:%S", d)

        return u"{0}".format(pval)