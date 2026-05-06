def run(self):
        '''runs 3dDeconvolve through the neural.utils.run shortcut'''
        out = nl.run(self.command_list(),products=self.prefix)
        if out and out.output:
            sds_list = re.findall(r'Stimulus: (.*?) *\n +h\[ 0\] norm\. std\. dev\. = +(\d+\.\d+)',out.output)
            self.stim_sds = {}
            for s in sds_list:
                self.stim_sds[s[0]] = float(s[1])