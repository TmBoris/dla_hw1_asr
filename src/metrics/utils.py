import editdistance
from collections import defaultdict

# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1
    return editdistance.eval(target_text, predicted_text) / len(target_text)
    

def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(target_text.split())


def expand_and_merge_path(dp, next_token_probs, ind2char, empty_tok):
    new_dp = defaultdict(float)
    for ind, next_token_prob in enumerate(next_token_probs):
        cur_char = ind2char[ind]
        for (prefix, last_char), v in dp.items():
            if last_char == cur_char:
                new_prefix = prefix
            else:
                if cur_char != empty_tok:
                    new_prefix = prefix + cur_char
                else:
                    new_prefix = prefix
            new_dp[(new_prefix, cur_char)] += v + next_token_prob
    return new_dp

def truncate_paths(dp, beam_size):
    return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])

def ctc_beam_search(probs, beam_size, ind2char, empty_tok):
    dp = {
        ('', empty_tok): 1.0,
    }
    for prob in probs:
        dp = expand_and_merge_path(dp, prob, ind2char, empty_tok)
        dp = truncate_paths(dp, beam_size)
    dp = [(prefix, proba) for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])]
    return dp[0][0]
