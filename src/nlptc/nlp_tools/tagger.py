class BMESTagger:
    def tag(self, sentence):
        chars = []
        tags = []
        for w in sentence:
            chars.extend(list(w))
            if len(w) == 1:
                tags.extend(['S'])
            else:
                tags.extend(['B'] + ['M'] * (len(w) - 2) + ['E'])
        return chars, tags

    def rebuild(self, chars, tags):
        assert len(chars) == len(tags)
        sentence = []
        w = ''
        b_offset = -1
        for i in range(len(chars)):
            if tags[i] == 'S':
                if w != '':
                    if b_offset == -1:
                        sentence.append(w)
                    else:
                        sentence.extend(chars[b_offset: i])
                        b_offset = -1
                    w = ''
                sentence.append(chars[i])

            if tags[i] == 'B':
                if w != '':
                    if b_offset == -1:
                        sentence.append(w)
                    else:
                        sentence.extend(chars[b_offset: i])
                w = chars[i]
                b_offset = i

            if tags[i] == 'M':
                if b_offset == -1:
                    sentence.append(chars[i])
                else:
                    w += chars[i]

            if tags[i] == 'E':
                if b_offset == -1:
                    sentence.append(chars[i])
                else:
                    w += chars[i]
                    sentence.append(w)
                    b_offset = -1
                    w = ''

        return sentence
