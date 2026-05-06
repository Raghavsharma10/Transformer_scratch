def _on_auth(self, user):
        """
        This is called when login with OpenID succeeded and it's not
        necessary to figure out if this is the users's first login or not.
        """
        app = current_app._get_current_object()
        if not user:
            # Google auth failed.
            login_error.send(app, user=None)
            abort(403)
        session["openid"] = user
        login.send(app, user=user)
        return redirect(request.args.get("next", None) or request.referrer or "/")