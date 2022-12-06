import sys, nltk
from evaluate import load
import pandas as pd
from utils import *
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


class TextSimplificationEvaluator:
    def __init__(self, reference, original, simplified):
        self.reference = reference
        self.original = original
        self.simplified = simplified
        self.sari = load('sari')

    def compute_metrics(self):
        assert len(self.reference) == len(self.original) == len(self.simplified)

        result = {'%SAME': 0, 'match_recall': [0, 0], 'match_precision': [0, 0],
                  'SARI': self.sari.compute(sources=self.original, predictions=self.simplified,
                                            references=[[reference] for reference in self.reference]), '#T/S': 0,
                  '#S/C': 0,
                  'bleu': corpus_bleu(list_of_references=[[reference] for reference in self.reference],
                                      hypotheses=self.simplified), '%MatchSS': 0}

        total_ss = 0

        for i in range(len(self.reference)):
            result['%SAME'] += 1 if self.original[i] == self.simplified[i] else 0

            ss = self.simplified[i].split(' | ')
            total_ss += len(ss)

            # #S/C
            result['#S/C'] += len(ss)

            # #T/S
            t_s = 0
            for s in ss:
                t_s += len(nltk.word_tokenize(s))
            result['#T/S'] += t_s

            # %Match
            rs = self.reference[i].split(' | ')
            if len(ss) == len(rs):
                result['%MatchSS'] += 1
            for s in ss:
                if s in rs:
                    result['match_recall'][0] += 1
                    result['match_precision'][0] += 1
            result['match_recall'][1] += len(rs)
            result['match_precision'][1] += len(ss)

        result['%MatchSS'] /= len(self.original)
        result['%SAME'] /= len(self.original)
        result['#S/C'] /= len(self.original)
        result['#T/S'] /= total_ss
        result['match_recall'] = result['match_recall'][0] / result['match_recall'][1]
        result['match_precision'] = result['match_precision'][0] / result['match_precision'][1]

        return result

    def compute_reference_metrics(self):
        result = {'%SAME': 0, '#T/S': 0, '#S/C': 0,
                  'SARI': self.sari.compute(sources=self.original, predictions=self.reference,
                                            references=[[reference] for reference in self.reference])}

        total_s = 0
        for i in range(len(self.reference)):
            result['%SAME'] += 1 if self.original[i] == self.reference[i] else 0
            rs = self.reference[i].split(' | ')
            total_s += len(rs)
            for s in rs:
                result['#T/S'] += len(nltk.word_tokenize(s))

            result['#S/C'] += len(rs)

        result['%SAME'] /= len(self.original)
        result['#S/C'] /= len(self.original)
        result['#T/S'] /= total_s

        return result
