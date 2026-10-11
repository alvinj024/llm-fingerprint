"""Text processing: tokenizer, vocabulary, task-type labelling, and the
text transformations used for the RQ4 ablations."""
import re
import string
import unicodedata
from collections import Counter

# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------
# Deliberately keeps case, punctuation, newlines and markdown markers as
# tokens: formatting is a plausible part of an LLM's "fingerprint", so we do
# not want the tokenizer to silently throw it away.
TOKEN_RE = re.compile(
    r"\n"                          # line break  -> <nl>
    r"|```"                        # code fence
    r"|\*\*"                       # bold marker
    r"|#{1,6}(?=\s)"               # markdown header marker
    r"|[A-Za-z]+(?:['’][A-Za-z]+)*"  # words (with apostrophes)
    r"|\d+"                        # numbers
    r"|[^\sA-Za-z\d]"              # any other single visible character
)

PAD, UNK, NL = "<pad>", "<unk>", "<nl>"


def tokenize(text: str):
    return [NL if t == "\n" else t for t in TOKEN_RE.findall(text)]


class Vocab:
    """Token <-> id mapping built from the TRAINING split only."""

    def __init__(self, token_lists, min_freq=2, max_size=30000):
        counts = Counter(t for toks in token_lists for t in toks)
        kept = [t for t, c in counts.most_common() if c >= min_freq][: max_size - 2]
        self.itos = [PAD, UNK] + kept
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len):
        ids = [self.stoi.get(t, 1) for t in tokens[:max_len]]
        return ids if ids else [1]  # never return an empty sequence


# --------------------------------------------------------------------------
# Language filter
# --------------------------------------------------------------------------
def mostly_english(text: str, threshold=0.9) -> bool:
    """True if >= `threshold` of the alphabetic characters are ASCII letters.
    A cheap, transparent proxy for 'written in English' that does not
    penalise code, numbers or emoji."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if c.isascii())
    return ascii_letters / len(letters) >= threshold


# --------------------------------------------------------------------------
# Task-type labelling of prompts (for RQ3)
# --------------------------------------------------------------------------
# Transparent keyword rules, applied in priority order coding > math >
# writing > general. Precision is checked by hand on a random sample and
# reported in the paper (see src/check_task_labels.py).
# "coding" = programming and technical computing (code, shells, OS, tools).
CODING_RE = re.compile(
    r"```|c\+\+|\bc#|\bdef \w+\(|\bimport \w+|"
    r"\b(python|javascript|typescript|java|golang|rust|sql|mysql|html|css|bash|"
    r"powershell|regex|json|yaml|api|function|code|coding|programming|"
    r"debug|compile|compiler|algorithm|pandas|numpy|pytorch|tensorflow|react|"
    r"django|flask|docker|kubernetes|git|linux|ubuntu|unity|arduino|perl|php|"
    r"ruby|kotlin|swift|scala|haskell|ocaml|matlab|cli|terminal|excel formula)\b",
    re.I,
)
# Arithmetic needs an explicit operator between numbers ("-" excluded: it is
# mostly ranges such as "24-54" or "7860 - 7880").
MATH_RE = re.compile(
    r"\b(math|maths|mathematics|mathematical|solve|equation|equations|integral|"
    r"derivative|calculus|probability|algebra|geometry|arithmetic|theorem|"
    r"prove|factorial|matrix|polynomial|logarithm|irrational|series)\b"
    r"|\d+\s*[\+\*/×^=]\s*\d+",
    re.I,
)
# Creative or composed writing: either a creative genre word, or a
# write/compose verb directly followed by a text type.
WRITING_RE = re.compile(
    r"\b(poem|poems|poetry|story|stories|essay|haiku|limerick|lyrics|song|"
    r"joke|jokes|screenplay|sonnet|rap|novel|fiction|fictional|tale|slogan|"
    r"tweet|roleplay|role-play|rewrite|rephrase|paraphrase|proofread)\b"
    r"|\b(write|compose|draft|create)\s+(me\s+)?(a|an|the|some|\d+)?\s*"
    r"([\w'-]+\s+){0,2}(letter|email|article|speech|post|message|paragraph|review|"
    r"script|bio|description|title|episode|toc|table of contents|ad|caption)s?\b",
    re.I,
)
TASKS = ["coding", "math", "writing", "general"]


def task_type(prompt: str) -> str:
    if CODING_RE.search(prompt):
        return "coding"
    if MATH_RE.search(prompt):
        return "math"
    if WRITING_RE.search(prompt):
        return "writing"
    return "general"


# --------------------------------------------------------------------------
# RQ4 ablation transforms (applied to BOTH training and test text, then
# the models are retrained from scratch)
# --------------------------------------------------------------------------
_MD_LINE_MARKERS = re.compile(r"^\s*(#{1,6}\s+|[-*•+]\s+|\d+[.)]\s+|>\s+)", re.M)


def strip_formatting(text: str) -> str:
    """Remove markdown and layout: code fences, bold/italic markers, header,
    bullet, numbered-list and quote markers, and all line breaks."""
    t = text.replace("```", " ")
    t = _MD_LINE_MARKERS.sub("", t)
    t = t.replace("**", "").replace("__", "")
    t = re.sub(r"\s+", " ", t)  # line breaks / indentation -> single spaces
    return t.strip()


def _is_punct(ch):
    return ch in string.punctuation or unicodedata.category(ch).startswith("P")


def lowercase_no_punct(text: str) -> str:
    """Lowercase and delete every punctuation character (layout kept)."""
    return "".join(c for c in text.lower() if not _is_punct(c))


def first_k_words(text: str, k=50) -> str:
    """Keep only the first k whitespace-separated words (layout inside them
    is preserved). Removes the response-length signal and anything that only
    appears later in a response (e.g. closing summaries)."""
    m = list(re.finditer(r"\S+", text))
    if len(m) <= k:
        return text
    return text[: m[k - 1].end()]


ABLATIONS = {
    "none": lambda t: t,
    "no_formatting": strip_formatting,
    "lower_no_punct": lowercase_no_punct,
    "first_50_words": first_k_words,
    "all_three": lambda t: lowercase_no_punct(strip_formatting(first_k_words(t))),
}
