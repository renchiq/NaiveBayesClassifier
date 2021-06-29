import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
from time import time
from numpy import prod


def reformat_text(text):
    # массив букв, удаляем из исходной строки знаки препинания
    nopunc = [char.lower() for char in text if char not in punctuation]
    nopunc = ''.join(nopunc)

    # массив слов из строки, где удалены нейтральные слова
    clean_words = [word for word in nopunc.split() if word not in stopwords.words('english')]
    return clean_words


# статистика употребления слов в спамах и хамах; 'word': [spam_count, ham_count]
def word_in_spam_and_ham():
    frequency_dict = {}
    for number, message in msg_train.items():
        for word in message:
            if word not in frequency_dict:
                frequency_dict[word] = [1, 0] if class_train[number] == 'spam' else [0, 1]
            else:
                if class_train[number] == 'spam':
                    frequency_dict[word][0] += 1
                if class_train[number] == 'ham':
                    frequency_dict[word][1] += 1
    return frequency_dict


# спамовость для каждого слова
def word_spaminess_calc():
    word_spaminess_data = {}
    word_frequency = word_in_spam_and_ham()
    spam_count = len([elem for elem in class_train.values if elem == 'spam'])
    ham_count = len([elem for elem in class_train.values if elem == 'ham'])
    for word, count in word_frequency.items():
        # if count == [1, 0] or count == [0, 1]:
        #     continue
        pr_w_s = count[0] / spam_count if count[0] != 0 else 0.001
        pr_s = spam_count / (spam_count + ham_count)
        pr_w_h = count[1] / ham_count if count[1] != 0 else 0.001
        pr_h = ham_count / (spam_count + ham_count)
        pr_s_w = (pr_w_s * pr_s) / (pr_w_s * pr_s + pr_w_h * pr_h)
        word_spaminess_data[word] = pr_s_w
    return word_spaminess_data


# определяем сообщение спам или нет
def spam_or_not():
    spaminess_info = word_spaminess_calc()
    classification_data = {}
    for number, message in msg_test.items():
        probabilities = []
        for word in message:
            if word in spaminess_info:
                probabilities.append(spaminess_info[word])
        if len(probabilities) != 0:
            total_probability = prod(probabilities) / (prod(probabilities) + prod([1 - p for p in probabilities]))
        else:
            total_probability = 0
        classification_data[number] = total_probability
    classification_data = {number: ('ham' if round(prob) == 0 else 'spam')
                           for number, prob in classification_data.items()}

    # вывод результатов
    right_results = len([1 for key in classification_data if classification_data[key] == class_test[key]])
    print('Our classifier`s accuracy: {:.3f}\n'.format(right_results / len(class_test) * 100))


start_time = time()

# читаем csv файл с нашим датасетом
messages = pd.read_csv('spamdb.csv', encoding='latin-1', sep=',')
# убираем ненужные столбцы
messages.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# разделение на обучающую и тестовую выборки
msg_train, msg_test, class_train, class_test = train_test_split(
    messages['text'], messages['class'], test_size=0.40, train_size=0.60)

# форматирование предложений, с заменой строк на списки
msg_train = {key: reformat_text(value) for key, value in msg_train.items()}
msg_test = {key: reformat_text(value) for key, value in msg_test.items()}

spam_or_not()

print('Execution time: {:.3f}s'.format(time() - start_time))
