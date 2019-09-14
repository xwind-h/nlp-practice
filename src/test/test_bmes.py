from nlptc.nlp_tools.tagger import BMESTagger

st = '这是  该  校  坚持  开展  素质  教育  的  成果  。'
chars = ['这', '是', '该', '校', '坚', '持', '开', '展', '素', '质', '教', '育', '的', '成', '果', '。']
tags = ['E', 'B', 'E', 'M', 'E', 'M', 'B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'S']

tagger = BMESTagger()
rst = tagger.rebuild(chars, tags)

print(st)
print(chars)
print(tags)
print(' '.join(rst))