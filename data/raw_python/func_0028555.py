def download(self, url, path, name_audio):
        """
            Params:

                ::url = Comprises the url used to download the audio.
                ::path =  Comprises the location where the file should be saved.
                ::name_audio = Is the name of the desired audio.
            
            Definition:

            Basically, we do a get with the requests module and after that 
            we recorded in the desired location by the developer or user, 
            depending on the occasion.
        """
        if path is not None:
            with open(str(path+name_audio), 'wb') as handle:
                response = requests.get(url, stream = True)
                if not response.ok:
                    raise Exception("Error in audio download.")
                for block in response.iter_content(1024):
                    if not block:
                        break
                    handle.write(block)