def delete_wiki(self, request, page, rev):
        """
        Deletes the page with all revisions or the revision, based on the
        users choice.

        Returns a HttpResponseRedirect.
        """

        # Delete the page
        if (
            self.cleaned_data.get('delete') == 'page'
            and request.user.has_perm('wakawaka.delete_revision')
            and request.user.has_perm('wakawaka.delete_wikipage')
        ):
            self._delete_page(page)
            messages.success(
                request, ugettext('The page %s was deleted' % page.slug)
            )
            return HttpResponseRedirect(reverse('wakawaka_index'))

        # Revision handling
        if self.cleaned_data.get('delete') == 'rev':

            revision_length = len(page.revisions.all())

            # Delete the revision if there are more than 1 and the user has permission
            if revision_length > 1 and request.user.has_perm(
                'wakawaka.delete_revision'
            ):
                self._delete_revision(rev)
                messages.success(
                    request,
                    ugettext('The revision for %s was deleted' % page.slug),
                )
                return HttpResponseRedirect(
                    reverse('wakawaka_page', kwargs={'slug': page.slug})
                )

            # Do not allow deleting the revision, if it's the only one and the user
            # has no permisson to delete the page.
            if revision_length <= 1 and not request.user.has_perm(
                'wakawaka.delete_wikipage'
            ):
                messages.error(
                    request,
                    ugettext(
                        'You can not delete this revison for %s because it\'s the '
                        'only one and you have no permission to delete the whole page.'
                        % page.slug
                    ),
                )
                return HttpResponseRedirect(
                    reverse('wakawaka_page', kwargs={'slug': page.slug})
                )

            # Delete the page and the revision if the user has both permissions
            if (
                revision_length <= 1
                and request.user.has_perm('wakawaka.delete_revision')
                and request.user.has_perm('wakawaka.delete_wikipage')
            ):
                self._delete_page(page)
                messages.success(
                    request,
                    ugettext(
                        'The page for %s was deleted because you deleted the only revision'
                        % page.slug
                    ),
                )
                return HttpResponseRedirect(reverse('wakawaka_index'))