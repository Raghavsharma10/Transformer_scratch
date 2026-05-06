def __search_email_by_subject(self, subject, match_recipient):
        "Get a list of message numbers"
        if match_recipient is None:
            _, data = self._mail.uid('search',
                                     None,
                                     '(HEADER SUBJECT "{subject}")'
                                     .format(subject=subject))

            uid_list = data[0].split()
            return uid_list
        else:
            _, data = self._mail.uid('search',
                                     None,
                                     '(HEADER SUBJECT "{subject}" TO "{recipient}")'
                                     .format(subject=subject, recipient=match_recipient))

            filtered_list = []
            uid_list = data[0].split()
            for uid in uid_list:
                # Those hard coded indexes [1][0][1] is a hard reference to the message email message headers
                # that's burried in all those wrapper objects that's associated
                # with fetching a message.
                to_addr = re.search(
                    "[^-]To: (.*)", self._mail.uid('fetch', uid, "(RFC822)")[1][0][1]).group(1).strip()

                if (to_addr == match_recipient or to_addr == "<{0}>".format(match_recipient)):
                    # Add matching entry to the list.
                    filtered_list.append(uid)

            return filtered_list