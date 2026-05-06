def login(self, request, form_class=None):
        "Login form"
        from django.contrib.auth import login as login_
        from django.contrib.auth.forms import AuthenticationForm

        if form_class is None:
            form_class = AuthenticationForm

        if request.POST:
            form = form_class(request, request.POST)
            if form.is_valid():
                login_(request, form.get_user())
                request.session.save()
                return HttpResponseRedirect(request.POST.get('next') or reverse('nexus:index', current_app=self.name))
            else:
                request.session.set_test_cookie()
        else:
            form = form_class(request)
            request.session.set_test_cookie()

        return self.render_to_response('nexus/login.html', {
            'form': form,
        }, request)