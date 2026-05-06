def print_commandless_help(self):
        """
        print_commandless_help
        """
        doc_help = self.m_doc.strip().split("\n")

        if len(doc_help) > 0:
            print("\033[33m--\033[0m")
            print("\033[34m" + doc_help[0] + "\033[0m")
            asp = "author  :"
            doc_help_rest = "\n".join(doc_help[1:])

            if asp in doc_help_rest:
                doc_help_rest = doc_help_rest.split("author  :")

                if len(doc_help_rest) > 1:
                    print("\n\033[33m" + doc_help_rest[0].strip() + "\n")
                    print("\033[37m" + asp + doc_help_rest[1] + "\033[0m")
                else:
                    print(doc_help_rest)
            else:
                print(doc_help_rest)

            print("\033[33m--\033[0m")
        else:
            print("\033[31mERROR, doc should have more then one line\033[0m")
            print(self.m_doc)