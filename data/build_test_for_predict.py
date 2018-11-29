import codecs

f_sentence = codecs.open("test.txt", encoding='utf-8').readlines()
fw = codecs.open("test.tsv", 'w', encoding='utf-8')
f_aspect = codecs.open("test_predict_aspect_ensemble.txt", encoding='utf-8').readlines()
test_texts = []
test_aspects = []
labels = []
fw.write("sentence\taspect\tlabel\n")
assert len(f_sentence) == len(f_aspect)
for line1, line2 in zip(f_sentence, f_aspect):
    splits = line1.strip('\n').split('\t')
    text = splits[0].strip()
    # text = text.lower()
    aspects = line2.strip('\n').split('|')
    if len(aspects) == 0:
        print('error for aspect reading')
        aspects = [None]
    for a in aspects:
        if a == '':
            print('error')
            a = '动力'
        test_texts.append(text)
        test_aspects.append(a)
assert len(test_texts) == len(test_aspects)

for t, a in zip(test_texts, test_aspects):
    fw.write(t + '\t' + a + '\t' + '0' + '\n')