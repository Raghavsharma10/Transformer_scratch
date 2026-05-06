def mark_thread_as_read(self, request, pk=None):
        """ Pk is the pk of the Thread to which the messages belong. """
        # we get the thread and check for permission
        thread = Thread.objects.get(id=pk)
        self.check_object_permissions(request, thread)
        # we save the date
        try:
            participation = Participation.objects.get(thread=thread, participant=request.rest_messaging_participant)
            participation.date_last_check = now()
            participation.save()
            # we return the thread
            serializer = self.get_serializer(thread)
            return Response(serializer.data)
        except Exception:
            return Response(status=status.HTTP_400_BAD_REQUEST)