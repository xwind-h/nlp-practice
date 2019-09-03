
def bmes_tagger(sentence):
    chars = []
    tags = []
    for w in sentence:
        chars.extend(list(w))
        if len(w) == 1:
            tags.extend(['S'])
        else:
            tags.extend(['B'] + ['M'] * (len(w) - 2) + ['E'])
    return chars, tags