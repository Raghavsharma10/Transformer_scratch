def dashboard(self, request):
        "Basic dashboard panel"
        # TODO: these should be ajax
        module_set = []
        for namespace, module in self.get_modules():
            home_url = module.get_home_url(request)

            if hasattr(module, 'render_on_dashboard'):
                # Show by default, unless a permission is required
                if not module.permission or request.user.has_perm(module.permission):
                    module_set.append((module.get_dashboard_title(), module.render_on_dashboard(request), home_url))

        return self.render_to_response('nexus/dashboard.html', {
            'module_set': module_set,
        }, request)