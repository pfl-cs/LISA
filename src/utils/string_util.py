# -*- coding: utf-8 -*-
import string
import re

def check_regex(pattern, s):
    match = pattern.match(s)

    return match is not None

def get_punctuation_pattern():
    punctuations = u'[\u2026' #
    ps = string.punctuation
    for p in ps:
        punctuations += '\\' + p
    punctuations += ']+$'

    pattern = re.compile(punctuations)
    return pattern

def is_punctuation(s):
    punctuations = u'[\u2026'  #
    ps = string.punctuation
    for p in ps:
        punctuations += '\\' + p
    punctuations += ']+$'
    return re.match(punctuations, s)

# a list like [1,2,3] to "1 2 3"
def list_to_str(l, sep=' ', use_unicode=False):
    if len(l) <= 0:
        return None
    if use_unicode:
        s = unicode(l[0])
        sep = unicode(sep.decode('utf-8'))
    else:
        s = str(l[0])
    for i in range(1, len(l)):
        item = l[i]
        if item is not None:
            if use_unicode:
                s = s + sep + unicode(item)
            else:
                s = s + sep + str(item)
    return s

# def set_to_str(items, sep=' '):
#     s = ''
#     for item in range(items):
#         if item is not None:
#             s = s + sep + str(item)
#     return s

# 如果第一个字母是#,则去掉
def remove_first_pound(s):
    if s[0] == '#':
        return s[1:]
    return s

if __name__ == '__main__':
    # pattern = get_punctuation_pattern()
    s = '#Se'
    # print check_regex(pattern, s)
    # print string.punctuation
    print is_punctuation(s)