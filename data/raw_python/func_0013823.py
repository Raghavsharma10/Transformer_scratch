def attach(self, lun_or_snap, skip_hlu_0=False):
        """ Attaches lun, snap or member snap of cg snap to host.

        Don't pass cg snapshot in as `lun_or_snap`.

        :param lun_or_snap: the lun, snap, or a member snap of cg snap
        :param skip_hlu_0: whether to skip hlu 0
        :return: the hlu number
        """

        # `UnityResourceAlreadyAttachedError` check was removed due to there
        # is a host cache existing in Cinder driver. If the lun was attached to
        # the host and the info was stored in the cache, wrong hlu would be
        # returned.
        # And attaching a lun to a host twice would success, if Cinder retry
        # triggers another attachment of same lun to the host, the cost would
        # be one more rest request of `modifyLun` and one for host instance
        # query.
        try:
            return self._attach_with_retry(lun_or_snap, skip_hlu_0)

        except ex.SystemAPINotSupported:
            # Attaching snap to host not support before 4.1.
            raise
        except ex.UnityAttachExceedLimitError:
            # The number of luns exceeds system limit
            raise
        except:  # noqa
            # other attach error, remove this lun if already attached
            self.detach(lun_or_snap)
            raise