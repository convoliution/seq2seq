def clean_word(word: str) -> str:
    return word.strip()                   \
               .lower()                   \
               .replace('’', '\'')        \
               .lstrip('<(\'"“')          \
               .rstrip('>)\'"”.,!?:;—-…')
