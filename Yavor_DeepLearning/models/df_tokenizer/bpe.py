#!/usr/bin/env python
# -*- coding:utf-8 -*-


def bep_merge(merges_file):
    bpe_merges = []
    with open(merges_file, encoding="utf-8") as merges_handle:
        for i, line in enumerate(merges_handle):
            line = line.strip()
            if (i == 0 and line.startswith("#version:")) or not line:
                continue
            bpe_merges.append(tuple(line.split()))
    bpe_rank = dict(zip(bpe_merges, range(len(bpe_merges))))
    return bpe_rank


bpe_ranks = bep_merge("bpe_ranks.txt")


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


cache = {}


def bpe(token):
    if token in cache:
        return cache[token]
    word = tuple(token)
    pairs = get_pairs(word)
    if not pairs:
        return token
    while True:
        bi_gram = min(pairs, key=lambda pair: bpe_ranks.get(pair, float("inf")))
        if bi_gram not in bpe_ranks:
            break
        first, second = bi_gram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                new_word.extend(word[i:])
                break
            else:
                new_word.extend(word[i:j])
                i = j

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    word = " ".join(word)
    cache[token] = word
    return word

