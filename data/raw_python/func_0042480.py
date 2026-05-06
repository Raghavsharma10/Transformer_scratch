def status_line(self):
        """
        Returns a status line for an item.

        Only really interesting when called for a draft
        item as it can tell you if the draft is the same as
        another version.
        """

        date = self.date_published
        status = self.state.title()
        if self.state == self.DRAFT:
            # Check if this item has changed since
            # our last publish
            status = "Draft saved"
            date = self.last_save
            if date and self.last_save == self.last_scheduled:
                # We need to figure out if the item it is based on
                # is either live now or will be live at some point.

                # If last_scheduled is less than or equal to
                # v_last_save this item is or will go live
                # at some point. Otherwise it won't
                # so we'll leave state as draft.
                if self.v_last_save:
                    if self.last_scheduled >= self.v_last_save:
                        status = self.PUBLISHED.title()

                    # The date this was scheduled is greater than
                    # what is currently live, this will go live at
                    # some point
                    if self.last_scheduled > self.v_last_save:
                        status = "Publish Scheduled"
                else:
                    status = "Publish Scheduled"

                date = self.date_published

        if date:
            status = "%s: %s" % (status, formats.date_format(date, "SHORT_DATE_FORMAT"))
        return status