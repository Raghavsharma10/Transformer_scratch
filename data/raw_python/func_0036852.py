def mark_deactivated(self,request,queryset):
        """An admin action for marking several cages as inactive.
		
        This action sets the selected cages as Active=False and Death=today.
        This admin action also shows as the output the number of mice sacrificed."""
        rows_updated = queryset.update(Active=False, End=datetime.date.today() )
        if rows_updated == 1:
            message_bit = "1 cage was"
        else:
            message_bit = "%s cages were" % rows_updated
        self.message_user(request, "%s successfully marked as deactivated." % message_bit)