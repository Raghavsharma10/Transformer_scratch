def ask(question, escape=True):
    "Return the answer"
    answer = raw_input(question)
    if escape:
        answer.replace('"', '\\"')
    return answer.decode('utf')