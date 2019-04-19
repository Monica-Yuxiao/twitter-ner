from ner.helper import start_with_capital, contains_special


def load_data(infpath):
    new_example = ([], [])
    with open(infpath) as inf:
        line = inf.readline()
        while line:
            # print(">", line.strip(), "<")
            if line.strip() != "":  # add to tuple
                word, tag = line.strip().split()
                new_example[0].append(word)
                new_example[1].append(tag)
            else:  # add current example to list, then start a new example
                examples.append(new_example)
                new_example = ([], [])
            line = inf.readline()
        examples.append(new_example)
    return examples


def add_features(examples):
    features = []
    for example in examples:
        sentence_feature = []
        for word in example[0]:
            word_feature = []
            word_feature.append(start_with_capital(word))
            word_feature.append(contains_special(word))
            # word_feature.append(no_alphabet(word))
            sentence_feature.append(word_feature)
        features.append(sentence_feature)
    return features


if __name__ == "__main__":
    # print(os.getcwd())
    # train_path = "../data/twitter_ner/train.txt"
    val_path = "../data/twitter_ner/validation.txt"

    # load training data
    examples = []
    examples = load_data(val_path)
    print("first example:\n", examples[0])
    print("last example:\n", examples[-1])

    features = add_features()
    print("feature for 1st example: \n", features[0])
    print("feature for last example: \n", features[-1])




