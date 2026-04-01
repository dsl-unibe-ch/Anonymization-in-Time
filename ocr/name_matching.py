"""
Name matching logic for the OCR pipeline.

Matches detected word-level boxes against a dictionary of names.
Features:
  - Length-aware fuzzy matching (short words require exact/near-exact match)
  - Common-word guard: frequent English/German words cannot be fuzzy-matched
  - Partial name support: "Amina" matches "Amina Hamzic" when only first name appears
  - Multi-word sequence matching across adjacent boxes on the same line
"""

import unicodedata
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Common words that must NOT fuzzy-match name parts.
# A word in this set can still match a name part via EXACT match.
# ---------------------------------------------------------------------------
COMMON_WORDS = {
    # English
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
    "its", "may", "new", "now", "old", "see", "two", "way", "who", "did",
    "its", "let", "put", "say", "she", "too", "use", "had", "man", "ago",
    "any", "ask", "boy", "did", "end", "few", "got", "had", "has", "hey",
    "him", "hot", "how", "ill", "its", "joy", "key", "lot", "low", "map",
    "met", "own", "pay", "per", "ran", "run", "sat", "set", "sir", "six",
    "ten", "yes", "yet",
    "also", "back", "been", "best", "both", "came", "come", "does", "done",
    "down", "each", "even", "ever", "find", "fine", "from", "gave", "give",
    "good", "great", "hand", "hard", "have", "here", "high", "home", "hope",
    "just", "keep", "kind", "know", "last", "left", "life", "like", "line",
    "live", "long", "look", "love", "made", "make", "many", "more", "most",
    "move", "much", "must", "name", "need", "next", "nice", "only", "open",
    "over", "past", "play", "real", "rest", "road", "room", "said", "same",
    "show", "side", "some", "soon", "stay", "such", "sure", "take", "talk",
    "than", "that", "them", "then", "there", "they", "this", "time", "told",
    "true", "turn", "type", "upon", "used", "very", "want", "well", "went",
    "were", "what", "when", "will", "with", "work", "year", "your",
    "about", "above", "after", "again", "along", "among", "begin", "being",
    "below", "bring", "built", "called", "carry", "close", "could", "doing",
    "every", "found", "given", "going", "group", "heard", "human", "large",
    "later", "light", "local", "might", "never", "night", "often", "other",
    "place", "point", "power", "quite", "ready", "right", "since", "small",
    "sound", "start", "state", "still", "story", "study", "taken", "their",
    "thing", "think", "those", "three", "today", "under", "until", "using",
    "where", "which", "while", "whole", "world", "would", "write", "wrote",
    "young",
    "please", "people", "little", "really", "should", "before", "always",
    "school", "family", "friend", "search", "message", "online", "status",
    "typing", "reply", "forward", "delete", "read", "send", "sent",
    "media", "photo", "video", "audio", "image", "file", "link", "group",
    "chat", "call", "voice", "block", "muted", "pinned", "archived",
    # German
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
    "und", "oder", "aber", "auch", "noch", "schon", "nur", "sehr",
    "mir", "mich", "dir", "dich", "sie", "wir", "ihr", "man",
    "ich", "ist", "bin", "hat", "war", "hat", "von", "mit", "bei",
    "aus", "auf", "an", "in", "zu", "im", "am", "zum", "zur",
    "nicht", "kein", "keine", "mehr", "wenn", "dann", "weil", "wie",
    "was", "wer", "wo", "wie", "dass", "ob", "ob", "bis", "nach",
    "seit", "uber", "unter", "neben", "zwischen", "vor", "hinter",
    "gut", "neu", "alt", "groß", "klein", "viel", "alle", "alles",
    "heute", "morgen", "jetzt", "schon", "noch", "immer", "wieder",
    "ok", "okay", "bitte", "danke", "hallo", "ja", "nein", "doch",
    "hey", "hi", "bye", "ciao",
}


def normalize(text: str) -> str:
    """Lowercase, strip diacritics and surrounding punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip().strip(".,;:!?()\"'-")


def _word_matches_name_part(word: str, name_part: str) -> tuple:
    """
    Check if a detected word matches a name part.

    Returns (matched: bool, confidence: float).

    Matching rules (most-restrictive-first):
      len <= 2 : exact match only
      len 3-4  : exact OR edit-distance == 1 AND not a common word
      len >= 5 : SequenceMatcher >= 0.85 AND not a common word (fuzzy only)

    Common-word guard: if the normalized OCR word is in COMMON_WORDS, only an
    exact match is accepted regardless of length.
    """
    nw = normalize(word)
    np_ = normalize(name_part)

    if not nw or not np_:
        return False, 0.0

    # Exact match always wins, regardless of common-word status
    if nw == np_:
        return True, 1.0

    # From here on, fuzzy/edit-distance is needed — block common words
    if nw in COMMON_WORDS:
        return False, 0.0

    max_len = max(len(nw), len(np_))

    if max_len <= 2:
        # No fuzzy for very short words — exact only (handled above)
        return False, 0.0

    if max_len <= 4:
        # Edit distance <= 1
        if abs(len(nw) - len(np_)) <= 1:
            diffs = sum(1 for a, b in zip(nw, np_) if a != b)
            diffs += abs(len(nw) - len(np_))
            if diffs <= 1:
                return True, 0.85
        return False, 0.0

    # len >= 5: fuzzy match with raised threshold
    ratio = SequenceMatcher(None, nw, np_).ratio()
    if ratio >= 0.85:
        return True, ratio
    return False, 0.0


def build_name_index(names_dict: dict) -> dict:
    """
    Pre-process the names dictionary into a lookup structure.

    Returns a dict:
      {
        "full_name_normalized": {
            "original": str,          # original key from dict
            "alterego": str,
            "words": [str, ...],      # normalized word parts
        },
        ...
      }
    """
    index = {}
    for name, alterego in names_dict.items():
        norm_name = " ".join(normalize(w) for w in name.strip().split())
        if not norm_name:
            continue
        words = norm_name.split()
        index[norm_name] = {
            "original": name,
            "alterego": alterego or "",
            "words": words,
        }
    return index


def _group_into_lines(word_boxes: list) -> list:
    """
    Group word boxes into horizontal lines.

    Returns list of lines, each line is a list of (original_idx, box) sorted
    left-to-right.  Lines are sorted top-to-bottom.
    """
    if not word_boxes:
        return []

    # Group by line_idx if present (from docTR), else fall back to y-center proximity
    if word_boxes[0].get("line_idx") is not None:
        line_map = {}
        for i, box in enumerate(word_boxes):
            lid = box["line_idx"]
            line_map.setdefault(lid, []).append((i, box))
        lines = list(line_map.values())
    else:
        # Fallback: cluster by y-center within 15px
        sorted_boxes = sorted(enumerate(word_boxes), key=lambda x: (x[1]["bbox"][1] + x[1]["bbox"][3]) / 2)
        lines = []
        current = [sorted_boxes[0]]
        for item in sorted_boxes[1:]:
            _, box = item
            prev_yc = (current[-1][1]["bbox"][1] + current[-1][1]["bbox"][3]) / 2
            curr_yc = (box["bbox"][1] + box["bbox"][3]) / 2
            if abs(curr_yc - prev_yc) < 15:
                current.append(item)
            else:
                lines.append(current)
                current = [item]
        lines.append(current)

    # Sort each line left-to-right, lines top-to-bottom
    for line in lines:
        line.sort(key=lambda x: x[1]["bbox"][0])
    lines.sort(key=lambda ln: min(b["bbox"][1] for _, b in ln))
    return lines


def _compute_parent_box(word_boxes: list) -> tuple:
    """Bounding hull of a list of word boxes."""
    x1 = min(b["bbox"][0] for b in word_boxes)
    y1 = min(b["bbox"][1] for b in word_boxes)
    x2 = max(b["bbox"][2] for b in word_boxes)
    y2 = max(b["bbox"][3] for b in word_boxes)
    return (x1, y1, x2, y2)


def _find_matches_in_line(line: list, name_index: dict) -> list:
    """
    Find all name matches in a single line of word boxes.

    Returns list of match dicts:
        {
            box_indices: set of original word-box indices consumed,
            bbox:        (x1, y1, x2, y2) merged box,
            text:        str  matched OCR text,
            name:        str  matched dictionary name,
            alterego:    str,
            confidence:  float,
            full_match:  bool,
        }
    """
    matches = []

    for norm_name, entry in name_index.items():
        name_words = entry["words"]
        alterego = entry["alterego"]
        alterego_words = alterego.split() if alterego else []
        n = len(name_words)

        for start in range(len(line)):
            matched_indices = []
            matched_boxes = []
            conf_scores = []
            word_ptr = 0
            box_ptr = start

            while word_ptr < n and box_ptr < len(line):
                orig_idx, box = line[box_ptr]
                box_text = box.get("text", "").strip()
                if not box_text:
                    box_ptr += 1
                    continue

                ok, conf = _word_matches_name_part(box_text, name_words[word_ptr])
                if ok:
                    matched_indices.append(orig_idx)
                    matched_boxes.append(box)
                    conf_scores.append(conf)
                    word_ptr += 1
                    box_ptr += 1
                else:
                    break

            if not matched_indices:
                continue

            avg_conf = sum(conf_scores) / len(conf_scores)
            is_full = (word_ptr == n)

            # Partial match rules: only allowed when
            # - name has 2+ words (single-word names require full match)
            # - matched portion has at least 4 characters
            if not is_full:
                if n < 2:
                    continue
                matched_text = " ".join(b.get("text", "") for b in matched_boxes)
                if len(normalize(matched_text)) < 4:
                    continue
                # Truncate alterego to matched word count
                partial_alterego = " ".join(alterego_words[:word_ptr]) if alterego_words else alterego
                avg_conf *= 0.95  # slight penalty vs full match
            else:
                partial_alterego = alterego

            ocr_text = " ".join(b.get("text", "") for b in matched_boxes)
            bbox = _compute_parent_box(matched_boxes)

            matches.append({
                "box_indices": set(matched_indices),
                "bbox": bbox,
                "text": ocr_text,
                "name": entry["original"],
                "alterego": partial_alterego,
                "confidence": avg_conf,
                "full_match": is_full,
            })

    if not matches:
        return []

    # Resolve overlaps: full > partial, longer > shorter, higher confidence
    matches.sort(
        key=lambda m: (m["full_match"], len(m["box_indices"]), m["confidence"]),
        reverse=True,
    )
    used = set()
    final = []
    for m in matches:
        if not m["box_indices"] & used:
            final.append(m)
            used |= m["box_indices"]
    return final


def filter_by_names(frame_boxes: dict, names_dict: dict) -> dict:
    """
    Match word-level boxes against names_dict and return filtered boxes.

    Each returned box has:
        bbox, text, confidence, line_idx   (from upstream)
        name:      matched dictionary name
        alterego:  replacement name
        parent_box: line bounding hull
        to_show:   True (all returned boxes are matches)
        track_id:  None (assigned later in stabilization)

    Boxes that do not match any name are dropped.

    Args:
        frame_boxes: {frame_idx: [word_box, ...]}
        names_dict:  {"Full Name": "Alterego"} dictionary

    Returns:
        {frame_idx: [matched_box, ...]}
    """
    name_index = build_name_index(names_dict)
    result = {}

    for frame_idx, boxes in frame_boxes.items():
        if not boxes:
            result[frame_idx] = []
            continue

        lines = _group_into_lines(boxes)
        # Precompute parent_box per line_idx
        line_parent = {}
        for line in lines:
            if not line:
                continue
            line_boxes_only = [b for _, b in line]
            pb = _compute_parent_box(line_boxes_only)
            for _, b in line:
                line_parent[id(b)] = pb

        frame_matches = []
        for line in lines:
            matches = _find_matches_in_line(line, name_index)
            for m in matches:
                # parent_box from the matched words' line
                matched_line_boxes = [b for _, b in line if _ in m["box_indices"]]
                pb = _compute_parent_box(matched_line_boxes) if matched_line_boxes else m["bbox"]

                frame_matches.append({
                    "bbox": m["bbox"],
                    "parent_box": pb,
                    "text": m["text"],
                    "name": m["name"],
                    "alterego": m["alterego"],
                    "confidence": m["confidence"],
                    "to_show": True,
                    "track_id": None,
                })

        result[frame_idx] = frame_matches

    return result
