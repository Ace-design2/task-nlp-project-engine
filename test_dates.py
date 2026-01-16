import dateparser

dates = [
    "friday",
    "this evening",
    "tonight"
]

replacements = {
    "this evening": "today 18:00",
    "tonight": "today 20:00",
    "morning": "today 09:00",
    "afternoon": "today 14:00"
}

print("Testing date parsing with settings and replacements...")
for d in dates:
    text_to_parse = replacements.get(d.lower(), d)
    parsed = dateparser.parse(text_to_parse, settings={'PREFER_DATES_FROM': 'future'})
    print(f"'{d}' (parsed as '{text_to_parse}') -> {parsed}")
