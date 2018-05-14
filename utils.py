def clean_word(word: str) -> str:
    return word.replace('’', '\'')       \
               .lstrip('(\'"“')          \
               .rstrip(')\'"”.,!?:;—-…')
