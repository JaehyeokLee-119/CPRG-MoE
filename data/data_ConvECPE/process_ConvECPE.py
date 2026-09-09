#!/usr/bin/env python3
"""Convert IEMOCAP emotion/cause annotations to the ConvECPE JSON format.

The source pickle is expected to contain the 12-item structure distributed with
the ConvECPE version of IEMOCAP. Only annotations and text are written; the
three feature collections in the pickle are intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import pickle
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable


EMOTIONS = {
    0: "happy",
    1: "sad",
    2: "neutral",
    3: "angry",
    4: "excited",
    5: "frustrated",
}
SPEAKERS = {"M": "A", "F": "B"}

REFERENCE_TRAIN_ORDER = [
    "Ses01F_script02_1", "Ses04M_impro04", "Ses02M_script01_1", "Ses04M_impro07",
    "Ses04M_script01_1", "Ses04M_script03_1", "Ses04M_script01_2", "Ses01M_impro01",
    "Ses02M_impro02", "Ses01F_impro05", "Ses04M_script03_2", "Ses02F_script03_2",
    "Ses02F_script02_2", "Ses03F_impro04", "Ses01M_script01_2", "Ses03M_impro07",
    "Ses03M_impro02", "Ses02M_script01_2", "Ses01M_impro03", "Ses02F_impro01",
    "Ses01M_impro04", "Ses04M_script02_2", "Ses01M_impro02", "Ses04M_impro05",
    "Ses01F_script02_2", "Ses02M_impro03", "Ses03M_script03_2", "Ses02M_script03_2",
    "Ses04F_script03_2", "Ses02M_impro07", "Ses03F_impro06", "Ses02M_impro05",
    "Ses04F_impro03", "Ses03M_impro04", "Ses02M_impro06", "Ses02F_impro06",
    "Ses04F_impro06", "Ses02F_script01_2", "Ses01M_script03_1", "Ses04F_impro02",
    "Ses04F_impro08", "Ses01F_impro02", "Ses03F_script02_1", "Ses02F_impro02",
    "Ses01F_script01_2", "Ses03F_impro08", "Ses01M_script02_1", "Ses03F_script01_2",
    "Ses02M_script03_1", "Ses03F_script03_1", "Ses03M_impro05b", "Ses01M_script01_3",
    "Ses03M_script02_2", "Ses04M_impro08", "Ses01F_impro04", "Ses01M_script02_2",
    "Ses04F_impro01", "Ses03M_impro08a", "Ses01F_impro03", "Ses01F_script03_1",
    "Ses03M_script01_3", "Ses01F_script01_3", "Ses03F_impro02", "Ses03F_script01_1",
    "Ses04F_impro05", "Ses03M_impro03", "Ses01M_impro07", "Ses02M_impro08",
    "Ses04M_impro02", "Ses04F_script03_1", "Ses04F_script01_2", "Ses02F_script02_1",
    "Ses02F_script01_3", "Ses04M_script01_3", "Ses04M_impro06", "Ses04F_script01_3",
    "Ses01M_script01_1", "Ses02F_impro03", "Ses01M_impro06", "Ses01F_script01_1",
    "Ses03F_impro07", "Ses01F_impro01", "Ses03F_impro03", "Ses03F_script01_3",
    "Ses04F_impro04", "Ses02F_impro05", "Ses04F_script02_2", "Ses03M_script01_2",
    "Ses01F_script03_2", "Ses03M_script02_1", "Ses03M_script03_1", "Ses02M_script01_3",
    "Ses04F_script01_1", "Ses04M_script02_1", "Ses03F_impro01", "Ses02F_script01_1",
    "Ses03F_script03_2", "Ses03M_script01_1", "Ses04F_script02_1", "Ses02M_impro04",
    "Ses03M_impro06", "Ses04M_impro01", "Ses02M_impro01", "Ses02F_script03_1",
    "Ses03M_impro05a", "Ses02F_impro07", "Ses03F_impro05", "Ses01F_impro07",
    "Ses04F_impro07", "Ses03M_impro08b", "Ses02F_impro08", "Ses02F_impro04",
    "Ses01M_impro05", "Ses04M_impro03", "Ses01M_script03_2", "Ses02M_script02_1",
    "Ses01F_impro06", "Ses03M_impro01", "Ses02M_script02_2", "Ses03F_script02_2",
]

REFERENCE_TEST_ORDER = [
    "Ses05M_script01_1b", "Ses05M_impro01", "Ses05M_impro04", "Ses05M_impro02",
    "Ses05F_impro01", "Ses05F_impro06", "Ses05F_script03_1", "Ses05M_impro08",
    "Ses05M_impro06", "Ses05M_script01_1", "Ses05F_impro05", "Ses05F_impro08",
    "Ses05M_impro05", "Ses05F_impro04", "Ses05M_script02_2", "Ses05F_script02_1",
    "Ses05F_impro07", "Ses05M_script02_1", "Ses05M_impro07", "Ses05M_script03_1",
    "Ses05F_impro03", "Ses05F_impro02", "Ses05F_script01_2", "Ses05F_script02_2",
    "Ses05M_script03_2", "Ses05F_script01_1", "Ses05M_impro03", "Ses05F_script01_3",
    "Ses05M_script01_3", "Ses05M_script01_2", "Ses05F_script03_2",
]


def partition(items: list[str], parts: int) -> list[list[str]]:
    """Split items into nearly equal consecutive parts."""
    quotient, remainder = divmod(len(items), parts)
    result: list[list[str]] = []
    start = 0
    for index in range(parts):
        size = quotient + (index < remainder)
        result.append(items[start : start + size])
        start += size
    return result


def ordered_ids(all_ids: Iterable[str], selected: set[str]) -> list[str]:
    """Return selected conversation IDs in stable source-dictionary order."""
    return [conversation_id for conversation_id in all_ids if conversation_id in selected]


def make_conversation(
    conversation_id: str,
    speakers: dict[str, list[str]],
    labels: dict[str, list[int]],
    cause_annotations: tuple[dict[str, list[int]], ...],
    sentences: dict[str, list[str]],
) -> list[list[dict[str, Any]]]:
    utterances = sentences[conversation_id]
    lengths = {
        len(utterances),
        len(speakers[conversation_id]),
        len(labels[conversation_id]),
        *(len(annotation[conversation_id]) for annotation in cause_annotations),
    }
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent field lengths for {conversation_id}: {sorted(lengths)}")

    turns: list[dict[str, Any]] = []
    for offset, utterance in enumerate(utterances):
        turn_number = offset + 1
        source_speaker = speakers[conversation_id][offset]
        emotion_label = labels[conversation_id][offset]
        try:
            speaker = SPEAKERS[source_speaker]
            emotion = EMOTIONS[emotion_label]
        except KeyError as error:
            raise ValueError(
                f"Unknown value {error.args[0]!r} in {conversation_id}, turn {turn_number}"
            ) from error

        turn: dict[str, Any] = {
            "turn": turn_number,
            "speaker": speaker,
            "utterance": utterance,
            "emotion": emotion,
        }

        causes = [annotation[conversation_id][offset] for annotation in cause_annotations]
        causes = [cause for cause in causes if cause != 0]
        if causes:
            spans: list[str] = []
            evidence: list[int | str] = []
            for cause in causes:
                if not isinstance(cause, Integral) or not 1 <= cause <= len(utterances):
                    raise ValueError(
                        f"Invalid cause {cause!r} in {conversation_id}, turn {turn_number}"
                    )
                cause = int(cause)
                spans.append(utterances[cause - 1])
                evidence.append("b" if cause > turn_number else cause)

            turn["expanded emotion cause span"] = spans
            turn["expanded emotion cause evidence"] = evidence
            turn["type"] = ["no_context"] * len(causes)

        turns.append(turn)

    return [turns]


def convert(source: Path, output_dir: Path, folds: int) -> None:
    with source.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, (list, tuple)) or len(payload) != 12:
        raise ValueError(f"Expected a 12-item pickle payload, got {type(payload).__name__}")

    (
        conversation_utterance_ids,
        speakers,
        labels,
        cause_1,
        cause_2,
        cause_3,
        _text_features,
        _visual_features,
        _audio_features,
        sentences,
        train_ids,
        test_ids,
    ) = payload

    source_order = list(conversation_utterance_ids)
    train_ids = set(train_ids)
    test_ids = set(test_ids)
    if train_ids & test_ids:
        raise ValueError("The predefined train and test conversation sets overlap")
    if train_ids | test_ids != set(source_order):
        raise ValueError("The predefined train/test sets do not cover all conversations")
    if folds < 2 or folds > len(train_ids):
        raise ValueError(f"folds must be between 2 and {len(train_ids)}")

    if folds != 5:
        raise ValueError("The published ConvECPE construction uses exactly 5 folds")
    if set(REFERENCE_TRAIN_ORDER) != train_ids or set(REFERENCE_TEST_ORDER) != test_ids:
        raise ValueError("The source pickle IDs do not match the published ConvECPE split manifest")
    validation_folds = partition(REFERENCE_TRAIN_ORDER, folds)
    ordered_test = REFERENCE_TEST_ORDER
    annotations = (cause_1, cause_2, cause_3)

    def build(ids: Iterable[str]) -> dict[str, list[list[dict[str, Any]]]]:
        return {
            conversation_id: make_conversation(
                conversation_id, speakers, labels, annotations, sentences
            )
            for conversation_id in ids
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    for fold, valid in enumerate(validation_folds):
        valid_set = set(valid)
        train = [item for item in REFERENCE_TRAIN_ORDER if item not in valid_set]
        split_ids = {"train": train, "valid": valid, "test": ordered_test}
        fold_dir = output_dir / f"data_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split, ids in split_ids.items():
            destination = fold_dir / f"ConvECPE_fold_{fold}_{split}.json"
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(build(ids), handle, ensure_ascii=False, indent="\t")
            print(f"wrote {destination} ({len(ids)} conversations)")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=script_dir.parent / "data_ConvECPE" / "IEMOCAP_emotion_cause_features.pkl",
        help="path to IEMOCAP_emotion_cause_features.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="directory in which data_<fold> directories are created",
    )
    parser.add_argument("--folds", type=int, default=5, help="number of validation folds")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    convert(arguments.source, arguments.output_dir, arguments.folds)
