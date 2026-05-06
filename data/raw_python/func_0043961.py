def load_raw_rules(cls, url):
        "Load raw rules from url or package file."
        raw_rules = []
        filename = url.split('/')[-1] # e.g.: easylist.txt
        try:
            with closing(request.get(url, stream=True)) as file:
                file.raise_for_status()
                # lines = 0 # to be removed
                for rule in file.iter_lines():
                    raw_rules.append(rule.strip())
                    # lines += 1 # tbr
                    # if lines == 2500: break # tbr, only for windoze with no re2
            logger.info("Adblock online %s: %d", filename, len(raw_rules))
        except: # file server down or bad url
            with open(resource_filename('summary', filename), 'r') as file:
                for rule in file:
                    raw_rules.append(rule.strip())
            logger.info("Adblock offline %s: %d", filename, len(raw_rules))
        return raw_rules