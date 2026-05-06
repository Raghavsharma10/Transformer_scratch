def resolve(cls, accept, available_renderers):
        """
        Resolves a list of accepted MediaTypes and available renderers to the preferred renderer.

        Call as MediaType.resolve([MediaType], [renderer]).
        """
        assert isinstance(available_renderers, tuple)
        accept = sorted(accept)

        renderers, seen = [], set()

        accept_groups = [[accept.pop()]]
        for imt in accept:
            if imt.equivalent(accept_groups[-1][0]):
                accept_groups[-1].append(imt)
            else:
                accept_groups.append([imt])

        for accept_group in accept_groups:
            for renderer in available_renderers:
                if renderer in seen:
                    continue
                for mimetype in renderer.mimetypes:
                    for imt in accept_group:
                        if mimetype.provides(imt):
                            renderers.append(renderer)
                            seen.add(renderer)
                            break

        return renderers