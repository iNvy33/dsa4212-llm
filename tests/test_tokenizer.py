import numpy as np

from src.tokenizer import CharTokenizer, encode_stories, train_bpe

TEXTS = [
    "Once upon a time, there was a cat.",
    "She said, “Let’s go!”\nThey went home.",
    "Café \U0001f34c",
]


def check_layout(tok, texts):
    tokens, offsets = encode_stories(tok, texts, batch=2)
    assert tokens.dtype == np.uint16 and offsets.dtype == np.int64
    assert len(offsets) == len(texts) + 1 and offsets[-1] == len(tokens) - 1
    assert (tokens[offsets] == tok.eot_id).all()
    assert (tokens == tok.eot_id).sum() == len(texts) + 1  # no EOT inside stories
    for i, text in enumerate(texts):
        story = tokens[offsets[i] + 1:offsets[i + 1]].tolist()
        assert tok.decode(story) == text


def test_char_tokenizer():
    tok = CharTokenizer()
    for t in TEXTS:
        assert tok.encode(t) == list(t.encode("utf-8"))
        assert tok.decode(tok.encode(t)) == t
    check_layout(tok, TEXTS)


def test_bpe_tokenizer():
    tok = train_bpe(TEXTS * 50, vocab_size=300)
    assert tok.vocab_size <= 300 and tok.eot_id is not None
    for t in TEXTS + ["unseen wörds — and \t tabs  "]:
        assert tok.decode(tok.encode(t)) == t  # byte-level BPE round-trips any text
    check_layout(tok, TEXTS)
