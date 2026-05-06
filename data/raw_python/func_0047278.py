def get_assessment_data(self, queryset, total_count, user_id):
        """
        Calculates and sets the following data for the supplied queryset:
            data = {
                'count': <the number of items in the queryset>
                'percentage': <percentage of total_count queryset represents>
                'is_user_call': <true if user made this call, false otherwise>
                'users': <set of all users who made this call>
            }
        """

        # We need to convert the usernames to strings here because the JSON
        # encoder will choke when serializing this data if the usernames are
        # unicode as they are when we get them back from the distinct call.
        users = [{'username': str(username), 'email': email}
                 for username, email
                 in queryset.values_list('user__username', 'user__email')
                            .distinct()]

        count = queryset.count()
        is_user_call = queryset.filter(user=user_id).exists()

        return {
            'count': count,
            'percentage': count / float(total_count) * 100.0,
            'is_user_call': is_user_call,
            'users': users,
        }