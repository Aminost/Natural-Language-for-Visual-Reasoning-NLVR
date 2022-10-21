import json

preprocessed_data = []
with open("preprocessed-dataset/preprocessed_train.json") as f:
    preprocessed_data = json.loads(f.read())

identified_words = []  # List of tuples (word, occurrences)
i = 0
for line in preprocessed_data:
    i += 1
    sentence = line["sentence"]
    for word in sentence:  # For each word of the sentence
        if len(identified_words) == 0:  # Just add the first word of the file
            identified_words.append((word, 1))
            continue
        # The following line returns [] if the word isn't in the identified_words list,
        # otherwise [(index, (word, occurrences))]
        checked_tuple = [(index, word_tuple) for index, word_tuple in enumerate(identified_words) if
                         word_tuple[0] == word]
        if not checked_tuple:
            identified_words.append((word, 1))
        else:
            checked_tuple = checked_tuple[0]  # getting the tuple out of the list
            identified_words[checked_tuple[0]] = (checked_tuple[1][0], checked_tuple[1][1] + 1)

    print(f"Sentence {i}/{len(preprocessed_data)} checked")

identified_words = sorted(identified_words, key=lambda t: t[1], reverse=True)

print(f"{len(identified_words)} words have been identified:")
print(identified_words)
