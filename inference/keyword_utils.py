"""
Keyword extraction + prompt-seeding utilities -- this is the "relevance to the streamer"
mechanism we decided on (see project.md "Context/relevance problem"): the model itself
has no context input, so relevance is faked at generation time by occasionally seeding
its prompt with a real keyword pulled from the streamer's recent transcript.

This module is deliberately dependency-free (no NLP libraries) -- it's a simple
stopword-filtered frequency extractor, good enough for short casual transcript lines.
"""

import re
import random

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because", "as", "of",
    "to", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from", "up", "down",
    "out", "off", "over", "under", "again", "further", "once", "is", "am", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "will", "would", "should", "could", "can", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "this", "that", "these", "those", "not", "no", "just", "really", "very",
    "like", "get", "got", "going", "gonna", "kinda", "actually", "okay", "ok", "yeah",
    "um", "uh", "well", "right", "now", "here", "there", "what", "which", "who", "whom",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "than", "too",
    # common contractions -- these slip past the length/stopword filter otherwise
    # (e.g. "let's" is 5 letters and looks like a real word, but it's pure filler)
    "let's", "lets", "that's", "it's", "i'm", "you're", "we're", "they're", "don't",
    "doesn't", "isn't", "wasn't", "didn't", "there's", "here's", "what's", "who's",
    "i've", "you've", "we've", "they've", "i'll", "you'll", "he's", "she's", "won't",
    "can't", "couldn't", "wouldn't", "shouldn't", "aren't", "weren't", "haven't",
    "hasn't", "hadn't",
}

WORD_RE = re.compile(r"[a-zA-Z']+")


def extract_keywords(text_or_lines, top_k=3):
    """Pulls up to top_k candidate keywords out of recent transcript text.

    Accepts either a single string or a list of recent lines (a rolling context window);
    if given a list, more recent lines aren't weighted specially -- kept simple on purpose.
    """
    if isinstance(text_or_lines, list):
        text = " ".join(text_or_lines)
    else:
        text = text_or_lines

    words = [w.lower() for w in WORD_RE.findall(text)]
    candidates = [w for w in words if len(w) > 3 and w not in STOPWORDS]

    if not candidates:
        return []

    # frequency count, but preserve first-seen order for ties (keeps it feeling natural
    # rather than alphabetical/random when everything only appears once)
    freq = {}
    order = []
    for w in candidates:
        if w not in freq:
            order.append(w)
        freq[w] = freq.get(w, 0) + 1

    ranked = sorted(order, key=lambda w: freq[w], reverse=True)
    return ranked[:top_k]


def build_seed_prompt(tokenizer, bos_id, mode_id, keyword=None, batch_size=1):
    """Builds a (batch_size, T) prompt tensor: <bos> <mode> [keyword tokens...].
    If keyword is None, the prompt is just <bos> <mode> -- pure unconditioned generation.
    """
    import torch

    ids = [bos_id, mode_id]
    if keyword:
        ids += tokenizer.encode(keyword).ids
    row = ids
    return torch.tensor([row] * batch_size, dtype=torch.long)


def pick_keyword_or_none(keywords, seed_probability=0.55):
    """Decides whether this batch of comments should be nudged toward a keyword at all --
    real chat doesn't react to every single word the streamer says, so this is
    deliberately probabilistic, not "always seed if a keyword exists"."""
    if keywords and random.random() < seed_probability:
        return random.choice(keywords)
    return None
