def create_tag(self, tag, message, sha, obj_type, tagger,
                   lightweight=False):
        """Create a tag in this repository.

        :param str tag: (required), name of the tag
        :param str message: (required), tag message
        :param str sha: (required), SHA of the git object this is tagging
        :param str obj_type: (required), type of object being tagged, e.g.,
            'commit', 'tree', 'blob'
        :param dict tagger: (required), containing the name, email of the
            tagger and the date it was tagged
        :param bool lightweight: (optional), if False, create an annotated
            tag, otherwise create a lightweight tag (a Reference).
        :returns: If lightweight == False: :class:`Tag <github3.git.Tag>` if
            successful, else None. If lightweight == True: :class:`Reference
            <github3.git.Reference>`
        """
        if lightweight and tag and sha:
            return self.create_ref('refs/tags/' + tag, sha)

        json = None
        if tag and message and sha and obj_type and len(tagger) == 3:
            data = {'tag': tag, 'message': message, 'object': sha,
                    'type': obj_type, 'tagger': tagger}
            url = self._build_url('git', 'tags', base_url=self._api)
            json = self._json(self._post(url, data=data), 201)
            if json:
                self.create_ref('refs/tags/' + tag, sha)
        return Tag(json) if json else None