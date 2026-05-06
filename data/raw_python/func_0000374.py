def run_question(self, question, input_func=_stdin_):
        """Run the given question."""
        qi = '[%d/%d] ' % (self.qcount, self.qtotal)
        print('%s %s:' % (qi, question['label']))
        while True:
            # ask for user input until we get a valid one
            ans = input_func('%s > ' % self.format_choices())
            if self.is_answer_valid(ans): 
                question['answer'] = int(ans)
                break
            else:
                if ans is '?': print(question['description'])
                else: print('Invalid answer.')
        self.qcount += 1