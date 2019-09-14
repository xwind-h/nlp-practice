def update(offset, index, terms):
    offset += len(terms[index])
    index += 1
    return offset, index


def calc_score(gold, predict):
    gold_offset = 0
    predict_offset = 0

    gold_term_index = 0
    predict_term_index = 0

    right = 0
    total = len(gold)
    right_and_wrong = len(predict)
    while gold_term_index < len(gold) or predict_term_index < len(predict):
        if gold_offset == predict_offset:
            if gold[gold_term_index] == predict[predict_term_index]:
                right += 1
            gold_offset, gold_term_index = update(gold_offset, gold_term_index, gold)
            predict_offset, predict_term_index = update(predict_offset, predict_term_index, predict)
        elif gold_offset < predict_offset:
            gold_offset, gold_term_index = update(gold_offset, gold_term_index, gold)
        else:
            predict_offset, predict_term_index = update(predict_offset, predict_term_index, predict)
    return right, total, right_and_wrong
