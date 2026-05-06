def logout(self, request):
        "Logs out user and redirects them to Nexus home"
        from django.contrib.auth import logout

        logout(request)

        return HttpResponseRedirect(reverse('nexus:index', current_app=self.name))