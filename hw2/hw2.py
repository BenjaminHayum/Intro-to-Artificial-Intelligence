import sys
import math


def get_parameter_vectors():
    """
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    described in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    """
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e = [0] * 26
    s = [0] * 26

    with open('e.txt', encoding='utf-8') as f:
        for line in f:
            # strip: removes the newline character
            # split: split the string on space character
            char, prob = line.strip().split(" ")
            # ord('E') gives the ASCII (integer) value of character 'E'
            # we then subtract it from 'A' to give array index
            # This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char) - ord('A')] = float(prob)
    f.close()

    with open('s.txt', encoding='utf-8') as f:
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char) - ord('A')] = float(prob)
    f.close()

    return e, s


def shred(filename):
    # Using a dictionary here. You may change this to any data structure of
    # your choice such as lists (X=[]) etc. for the assignment
    letter_count_dictionary = dict()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            for character in line:
                uppercase_char = character.upper()
                ascii_val = ord(uppercase_char)
                if 90 >= ascii_val >= 65:
                    if uppercase_char in letter_count_dictionary.keys():
                        letter_count_dictionary[uppercase_char] += 1
                    else:
                        letter_count_dictionary[uppercase_char] = 1
    return letter_count_dictionary


def x_log_lang(letter, filename):
    letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                       'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
                       'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

    letter_count_dictionary = shred(filename)
    # Get parameters returns a list, not a dictionary
    x_english_list, x_spanish_list = get_parameter_vectors()

    list_index = letter_to_index[letter]
    if letter in letter_count_dictionary.keys():
        # I formerly rounded these in the past submission which created some errors!!!
        # X_1 * log(e_1)
        english_prob = letter_count_dictionary[letter] * math.log(x_english_list[list_index])
        # X_1 * log(s_1)
        spanish_prob = letter_count_dictionary[letter] * math.log(x_spanish_list[list_index])
        return english_prob, spanish_prob

    else:
        return -0.0000, -0.0000


def function_y(language, filename):
    all_letters = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z'
    ]
    sum_letter_probs = 0
    if language == 'english':
        for letter in all_letters:
            english_prob, _ = x_log_lang(letter, filename)
            sum_letter_probs += english_prob
        p_english = 0.6
        return math.log(p_english) + sum_letter_probs

    elif language == 'spanish':
        for letter in all_letters:
            _, spanish_prob = x_log_lang(letter, filename)
            sum_letter_probs += spanish_prob
        p_spanish = 0.4
        return math.log(p_spanish) + sum_letter_probs


def p_lang_given_x(filename):
    f_spanish = function_y('spanish', filename)
    f_english = function_y('english', filename)

    if f_spanish - f_english >= 100:
        p_english_given_x = 0
    elif f_spanish - f_english <= -100:
        p_english_given_x = 1
    else:
        p_english_given_x = 1 / (1 + math.exp(f_spanish - f_english))
    return p_english_given_x


#
#
#
#
#
#
#
#
# Calling all of the above functions and answering the questions!

fileName = 'samples/letter2.txt'

# Q1
all_letters = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
 ]
print('Q1')
letter_count_dictionary = shred(fileName)
for letter in all_letters:
    if letter in letter_count_dictionary.keys():
        count = letter_count_dictionary[letter]
    else:
        count = 0
    print(letter + " " + str(count))

# Q2
print('Q2')
question_letter = 'A'
output = x_log_lang(question_letter, fileName)
print(format(output[0], ".4f"))
print(format(output[1], ".4f"))

# Q3
print('Q3')
f_english_ = function_y('english', fileName)
print(format(f_english_, ".4f"))
f_spanish_ = function_y('spanish', fileName)
print(format(f_spanish_, ".4f"))

# Q4
print('Q4')
print(format(p_lang_given_x(fileName), ".4f"))

