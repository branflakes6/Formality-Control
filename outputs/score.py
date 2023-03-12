
import argparse
from collections import defaultdict
import re
from typing import List, Pattern, Tuple

FORMALITY_PHRASES = re.compile("(\[F\](.*?)\[/F\])")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-hyp",
        "--hypotheses",
        type=str,
        help="File containing system detokenized output translations"
    )
    parser.add_argument(
        "-f",
        "--formal_refs",
        type=str,
        help="File containing formal references with annotated grammatical formality"
    )
    parser.add_argument(
        "-if",
        "--informal_refs",
        type=str,
        help="File containing informal references with annotated grammatical formality."
    )
    parser.add_argument(
        "-nd",
        "--non_whitespace_delimited",
        action="store_true",
        help="If the target language tokens are non-whitespace delimited (e.g. for Japanese)"
    )

    return parser.parse_args()

def compute_score(
    hypotheses: str,
    annotated_formal_refs: str,
    annotated_informal_refs: str,
    tok_split: bool=True
) -> Tuple[float, float]:

    hypotheses = _read_lines(hypotheses)
    annotated_references_formal = _read_lines(annotated_formal_refs)
    annotated_references_informal = _read_lines(annotated_informal_refs)

    if not (len(hypotheses) == len(annotated_references_formal) == len(annotated_references_informal) > 0):
        raise RuntimeError("Empty or mismatched hypotheses and reference files.")

    scores = defaultdict(int)
    for hyp, ref_formal, ref_informal in zip(hypotheses, annotated_references_formal, annotated_references_informal):
        formal_phrases = get_matching_phrases(hyp, ref_formal, tok_split, FORMALITY_PHRASES)
        informal_phrases = get_matching_phrases(hyp, ref_informal, tok_split, FORMALITY_PHRASES)

        label = predict_formality_label(formal_phrases, informal_phrases)
        
        scores[f"ref_matched_count_{label}"] += 1

    n_matched = scores["ref_matched_count_INFORMAL"] + scores["ref_matched_count_FORMAL"]
    formal_acc = scores["ref_matched_count_FORMAL"]/n_matched if n_matched>0 else 0
    informal_acc = scores["ref_matched_count_INFORMAL"]/n_matched if n_matched>0 else 0

    return formal_acc, informal_acc


def get_matching_phrases(
    hyp: str,
    anno_ref: str,
    tok_split: bool=True,
    phrase_regex: Pattern=FORMALITY_PHRASES
):

    anno_ph = re.findall(phrase_regex, anno_ref)
    if not tok_split:
        anno_ph_hyp = [ph for _, ph in anno_ph if ph in hyp]
    else:
        anno_ph_hyp = [ph for _, ph in anno_ph if set(ph.split(" ")).issubset(hyp.split(" "))]
    return anno_ph_hyp


def predict_formality_label(
    ph_formal_hyp: List[str],
    ph_informal_hyp: List[str]
) -> str:

    if ph_formal_hyp and not ph_informal_hyp:
        return "FORMAL"
    elif ph_informal_hyp and not ph_formal_hyp:
        return "INFORMAL"
    elif not ph_informal_hyp and not ph_formal_hyp:
        return "NEUTRAL"
    return "OTHER"

def _read_lines(file: str) -> List[str]:
    with open(file, "r", encoding="utf-8") as file:
        raw_text = [line.strip() for line in file.readlines()]

    return raw_text


if __name__ == "__main__":
    args = parse_args()
    whitespace_delimited_tokens = not args.non_whitespace_delimited

    formal_acc, informal_acc = compute_score(
        args.hypotheses, args.formal_refs, args.informal_refs, tok_split=whitespace_delimited_tokens
    ) 
    print(f"Formal Acc: {formal_acc:.3f}, Informal Acc: {informal_acc:.3f}")
