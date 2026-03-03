# src/utils/names.py
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from typing import Dict, List, Optional, Literal, Tuple


Status = Literal[
    "missing",
    "exact",
    "norm_fuzzy",
    "lat_exact",
    "lat_fuzzy",
    "fallback",
    "unknown"]


_RU2LAT: Dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya"}


def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _strip_decorations(name: str) -> str:
    s = _collapse_spaces(str(name or ""))
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    s = re.sub(r"\(.*?\)", "", s).strip()
    return _collapse_spaces(s)


def _norm_key(s: str) -> str:
    s = _strip_decorations(s).casefold()
    s = s.replace("ё", "е")
    return _collapse_spaces(s)


def _ru_to_lat_key(s: str) -> str:
    s = _strip_decorations(s).casefold().replace("ё", "е")
    out: List[str] = []
    for ch in s:
        if ch in _RU2LAT:
            out.append(_RU2LAT[ch])
        elif "a" <= ch <= "z" or "0" <= ch <= "9":
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
    return _collapse_spaces("".join(out))


def _ratio(a: str, b: str) -> float:
    return float(SequenceMatcher(None, a, b).ratio())


@dataclass
class CanonicalizeResult:
    input: Optional[str]
    output: Optional[str]
    status: Status
    score: Optional[float] = None
    detail: str = ""


class NameCanonicalizer:
    def __init__(self, characters: List[str], locations: List[str]):
        self.characters = [c for c in (characters or []) if (c or "").strip()]
        self.locations = [l for l in (locations or []) if (l or "").strip()]
        self._char_by_norm, self._char_by_lat = self._build_maps(self.characters)
        self._loc_by_norm, self._loc_by_lat = self._build_maps(self.locations)

    @staticmethod
    def _build_maps(canon_list: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        by_norm: Dict[str, str] = {}
        by_lat: Dict[str, List[str]] = {}
        for c in canon_list:
            by_norm[_norm_key(c)] = c
            lk = _ru_to_lat_key(c)
            by_lat.setdefault(lk, []).append(c)
        return by_norm, by_lat

    def canonicalize_character(
        self,
        name: Optional[str],
        *,
        fallback: Optional[str] = None,
        cutoff: float = 0.88,
    ) -> CanonicalizeResult:
        return self._canonicalize(name, self._char_by_norm, self._char_by_lat, fallback=fallback, cutoff=cutoff)

    def canonicalize_location(
        self,
        name: Optional[str],
        *,
        fallback: Optional[str] = None,
        cutoff: float = 0.88,
    ) -> CanonicalizeResult:
        return self._canonicalize(name, self._loc_by_norm, self._loc_by_lat, fallback=fallback, cutoff=cutoff)

    @staticmethod
    def _canonicalize(
        name: Optional[str],
        by_norm: Dict[str, str],
        by_lat: Dict[str, List[str]],
        *,
        fallback: Optional[str],
        cutoff: float,
    ) -> CanonicalizeResult:
        if name is None:
            return CanonicalizeResult(input=None, output=fallback if fallback else None, status="missing")

        raw = _strip_decorations(str(name))
        if not raw:
            if fallback:
                return CanonicalizeResult(input=name, output=fallback, status="fallback", detail="empty_input")
            return CanonicalizeResult(input=name, output=None, status="missing")

        nk = _norm_key(raw)
        if nk in by_norm:
            return CanonicalizeResult(input=name, output=by_norm[nk], status="exact", score=1.0)

        norm_keys = list(by_norm.keys())
        near_norm = get_close_matches(nk, norm_keys, n=1, cutoff=cutoff)
        if near_norm:
            out = by_norm[near_norm[0]]
            return CanonicalizeResult(input=name, output=out, status="norm_fuzzy", score=_ratio(nk, near_norm[0]), detail="fuzzy_norm")

        lk = _ru_to_lat_key(raw)
        if lk in by_lat:
            cands = by_lat[lk]
            if len(cands) == 1:
                return CanonicalizeResult(input=name, output=cands[0], status="lat_exact", score=1.0)
            best = sorted(((_ratio(raw, c), c) for c in cands), reverse=True)[0][1]
            return CanonicalizeResult(input=name, output=best, status="lat_exact", detail="ambiguous_lat")

        lat_keys = list(by_lat.keys())
        near_lat = get_close_matches(lk, lat_keys, n=1, cutoff=cutoff)
        if near_lat:
            key = near_lat[0]
            cands = by_lat.get(key) or []
            out = cands[0] if cands else None
            if out:
                return CanonicalizeResult(input=name, output=out, status="lat_fuzzy", score=_ratio(lk, key), detail="fuzzy_lat")

        if fallback:
            return CanonicalizeResult(input=name, output=fallback, status="fallback", detail="no_match_used_fallback")

        return CanonicalizeResult(input=name, output=None, status="unknown", detail="no_match")