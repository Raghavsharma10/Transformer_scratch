def mark_sacrificed(self,request,queryset):
        """An admin action for marking several animals as sacrificed.
		
        This action sets the selected animals as Alive=False, Death=today and Cause_of_Death as sacrificed.  To use other paramters, mice muse be individually marked as sacrificed.
        This admin action also shows as the output the number of mice sacrificed."""
        rows_updated = queryset.update(Alive=False, Death=datetime.date.today(), Cause_of_Death='Sacrificed')
        if rows_updated == 1:
            message_bit = "1 animal was"
        else:
            message_bit = "%s animals were" % rows_updated
        self.message_user(request, "%s successfully marked as sacrificed." % message_bit)