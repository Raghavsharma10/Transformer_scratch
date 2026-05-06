def _build_instance_handler_mapping(cls, instance, handle_d):
        """For every unbound handler, get the bound version."""
        res = {}
        for member_name, sig_name in handle_d.items():
            if sig_name in res:
                sig_handlers = res[sig_name]
            else:
                sig_handlers = res[sig_name] = []
            sig_handlers.append(getattr(instance, member_name))
        return res