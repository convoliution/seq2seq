def clean_word(word: str) -> str:
    return word.strip()                   \
               .replace('’', '\'')        \
               .lstrip('<(\'"“')          \
               .rstrip('>)\'"”.,!?:;—-…')
