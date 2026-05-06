def request_connect(self, act, coro):
        "Requests a connect for `coro` corutine with parameters and completion \
        passed via `act`"
        result = self.try_run_act(act, perform_connect)
        if result:
            return result, coro
        else:
            self.add_token(act, coro, perform_connect)