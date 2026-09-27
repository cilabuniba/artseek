import re
import string
from typing import Any, List, Literal

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.collections import MetricCollection
from torchmetrics.text import BERTScore, BLEUScore, ROUGEScore
from torchmetrics.utilities.data import _flatten_dict

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class Evaluator:
    """COCO caption metrics (pycocoevalcap): BLEU@1-4, METEOR, ROUGE-L, CIDEr
    and, optionally, SPICE. METEOR and SPICE need Java."""

    def __init__(self, with_spice: bool = True) -> None:
        self.tokenizer = PTBTokenizer()
        self.scorer_list = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
        if with_spice:
            self.scorer_list.append((Spice(), "SPICE"))
        self.evaluation_report = {}

    def do_the_thing(self, golden_reference, candidate_reference):
        golden_reference_dict = []
        for refs in golden_reference:
            refs_dict = [{"caption": ref} for ref in refs]
            golden_reference_dict.append(refs_dict)
        golden_reference = {k: v for k, v in enumerate(golden_reference_dict)}
        candidate_reference = {k: [{'caption': v}] for k, v in enumerate(candidate_reference)}
        
        golden_reference = self.tokenizer.tokenize(golden_reference)
        candidate_reference = self.tokenizer.tokenize(candidate_reference)
        
        # As in https://github.com/salaniz/pycocoevalcap/blob/a24f74c408c918f1f4ec34e9514bc8a76ce41ffd/eval.py#L51-L63
        for scorer, method in self.scorer_list:
            score, scores = scorer.compute_score(golden_reference, candidate_reference)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.evaluation_report[m] = sc
            else:
                self.evaluation_report[method] = score

class RobustAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "accuracies",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _preprocess(self, s: str) -> str:
        # 1. Make all characters lowercase.
        s = s.lower()

        # 2. Remove periods except if it occurs as a decimal separator.
        #    We remove any period that is not flanked on both sides by a digit.
        s = re.sub(r"(?<!\d)\.(?!\d)", "", s)

        # 3. Convert number words to digits.
        #    (This simple mapping only covers common number words.)
        number_words = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
            "thirty": "30",
            "forty": "40",
            "fifty": "50",
            "sixty": "60",
            "seventy": "70",
            "eighty": "80",
            "ninety": "90",
            "hundred": "100",
            "thousand": "1000",
        }
        # Replace any occurrence of these words as whole words.
        pattern = (
            r"\b(" + "|".join(re.escape(word) for word in number_words.keys()) + r")\b"
        )
        s = re.sub(pattern, lambda m: number_words[m.group(0)], s)

        # 4. Remove articles (a, an, the).
        s = re.sub(r"\b(a|an|the)\b", "", s)

        # 5. Add apostrophes to common contractions missing them.
        #    (This mapping is simplistic and can be extended.)
        contractions = {
            "dont": "don't",
            "cant": "can't",
            "wont": "won't",
            "im": "i'm",
            "ive": "i've",
            "id": "i'd",
            "youre": "you're",
            "youve": "you've",
            "hes": "he's",
            "shes": "she's",
            "isnt": "isn't",
            "arent": "aren't",
            "wasnt": "wasn't",
            "werent": "weren't",
            "didnt": "didn't",
            "hasnt": "hasn't",
            "havent": "haven't",
            "wouldnt": "wouldn't",
            "couldnt": "couldn't",
            "shouldnt": "shouldn't",
            "lets": "let's",
        }
        # Replace each contraction (when it appears as a separate word).
        for wrong, correct in contractions.items():
            s = re.sub(r"\b" + wrong + r"\b", correct, s)

        # 6. Replace punctuation (except apostrophe and colon) with a space.
        #    Special handling for commas: if a comma is found between digits (e.g. "100,978")
        #    then remove it without inserting a space.
        s = re.sub(r"(?<=\d),(?=\d)", "", s)

        # Build a string of punctuation characters to replace.
        # Exclude apostrophe (') and colon (:) and comma (already handled).
        punctuation_to_replace = "".join(
            ch for ch in string.punctuation if ch not in {"'", ":", ","}
        )
        s = re.sub(f"[{re.escape(punctuation_to_replace)}]", " ", s)

        # Finally, collapse multiple whitespace characters into a single space and trim.
        s = re.sub(r"\s+", " ", s).strip()

        return s

    def _input_format(self, preds: List[str], target: List[List[str]]) -> torch.Tensor:
        # Preprocess predictions and targets
        preds = [self._preprocess(p) for p in preds]
        targets = [[self._preprocess(t) for t in ts] for ts in target]

        # Calculate human agreement metric
        results = []
        for pred, golds in zip(preds, targets):
            matches = sum(1 for gold in golds if pred == gold)
            results.append(min(matches / min(len(golds), 3), 1))
        return torch.tensor(results)

    def update(self, preds: List[str], target: List[List[str]]) -> None:
        accuracies = self._input_format(preds, target)
        self.accuracies += accuracies.sum()
        self.total += len(accuracies)

    def compute(self) -> torch.Tensor:
        return self.accuracies / self.total


class VQAMetrics(MetricCollection):
    def __init__(self, with_bert_score=True, **kwargs):
        metrics = {
            "robust_accuracy": RobustAccuracy(),
            "bleu1_score": BLEUScore(n_gram=1),
            "bleu4_score": BLEUScore(n_gram=4),
            "rouge_score": ROUGEScore(),
        }
        if with_bert_score:
            metrics["bert_score"] = BERTScore(model_name_or_path="roberta-large")
        super().__init__(
            metrics,
            **kwargs,
        )
        
    def _compute_and_reduce(
        self, method_name: Literal["compute", "forward"], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Compute result from collection and reduce into a single dictionary.

        Args:
            method_name: The method to call on each metric in the collection.
                Should be either `compute` or `forward`.
            args: Positional arguments to pass to each metric (if method_name is `forward`)
            kwargs: Keyword arguments to pass to each metric (if method_name is `forward`)

        Raises:
            ValueError:
                If method_name is not `compute` or `forward`.

        """
        result = {}
        for k, m in self.items(keep_base=True, copy_state=False):
            if method_name == "compute":
                res = m.compute()
            elif method_name == "forward":
                # if m is the bert score metric, only pass the first target
                if k == "bert_score":
                    bert_score_args = list(args)
                    bert_score_kwargs = kwargs.copy()
                    if len(args) > 1:
                        bert_score_target = [t[0] for t in args[1]]
                        bert_score_args = (args[0], bert_score_target)
                    else:
                        bert_score_target = [t[0] for t in kwargs.get("target", None)]
                        bert_score_kwargs["target"] = bert_score_target
                    res = m(*bert_score_args, **m._filter_kwargs(**bert_score_kwargs))
                else:
                    res = m(*args, **m._filter_kwargs(**kwargs))
            else:
                raise ValueError(f"method_name should be either 'compute' or 'forward', but got {method_name}")
            result[k] = res

        _, duplicates = _flatten_dict(result)

        flattened_results = {}
        for k, m in self.items(keep_base=True, copy_state=False):
            res = result[k]
            if isinstance(res, dict):
                for key, v in res.items():
                    # if duplicates of keys we need to add unique prefix to each key
                    if duplicates:
                        stripped_k = k.replace(getattr(m, "prefix", ""), "")
                        stripped_k = stripped_k.replace(getattr(m, "postfix", ""), "")
                        key = f"{stripped_k}_{key}"
                    if getattr(m, "_from_collection", None) and m.prefix is not None:
                        key = f"{m.prefix}{key}"
                    if getattr(m, "_from_collection", None) and m.postfix is not None:
                        key = f"{key}{m.postfix}"
                    flattened_results[key] = v
            else:
                flattened_results[k] = res
        return {self._set_name(k): v for k, v in flattened_results.items()}
        
        
# vqa_metrics = VQAMetrics(with_bert_score=False)
# print(vqa_metrics(["ciao"], [["ciao"]]))
