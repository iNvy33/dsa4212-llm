from src.cbt import detokenize, detokenize_line


def test_punctuation_and_clitics():
    assert detokenize_line("We will hide it in a corner , and we wo n't touch it .") == \
        "We will hide it in a corner, and we won't touch it."
    assert detokenize_line("the king 's aunts were old-fashioned ; they 're cats !") == \
        "the king's aunts were old-fashioned; they're cats!"


def test_quotes_and_brackets():
    assert detokenize_line("`` They are kind , '' said the king .") == '"They are kind," said the king.'
    assert detokenize_line("` But we must provide , ' said the Cat .") == "'But we must provide,' said the Cat."
    assert detokenize_line("say : `` HER daughter shall marry . ''") == 'say: "HER daughter shall marry."'
    assert detokenize_line("a cat -LRB- a big one -RRB- sat") == "a cat (a big one) sat"
    assert detokenize_line("the girls ' dolls") == "the girls' dolls"  # plural possessive


def test_sentences_joined_with_spaces():
    assert detokenize("One .\nTwo !\n\nThree ?") == "One. Two! Three?"
