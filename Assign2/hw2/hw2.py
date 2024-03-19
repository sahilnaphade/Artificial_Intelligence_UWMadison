import sys
import math
from math import log


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    STD_CHAR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for each_char in STD_CHAR:
        X[each_char] = 0
    with open (filename,encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(" ")
            # print(line)
            for each_word in line:
                for each_char in each_word:
                    if each_char.upper() in STD_CHAR:
                        ch = each_char.upper()
                        if ch.isalpha():
                            if ch in X:
                                X[ch] += 1
                            else:
                                X[ch] = 1
    return X


# Sorted_q1 = observed count, letter_prob_vec is the known probability
def sum_observed_log_prob(observed_sorted_freq, letter_prob_vec):
    word_count = list(observed_sorted_freq.items())
    # print(word_count)
    total_sum = 0.0
    for iter in range(0, len(letter_prob_vec)):
        # print("Multiplying {} and log value of letter_prob {} -> {}".format(word_count[iter][1], letter_prob_vec[iter], log(letter_prob_vec[iter])))
        total_sum += (word_count[iter][1] * log(letter_prob_vec[iter]))
        # print(total_sum)
    return total_sum


# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

if __name__ == "__main__":
    blind_english_prob = 0.6
    blind_spanish_prob = 1 - blind_english_prob
    
    filename = "letter.txt"
    
    print("Q1")
    q1_sol = shred(filename)
    observed_sorted_freq = dict(sorted(q1_sol.items()))
    for key, item in observed_sorted_freq.items():
        print(f"{key} {item}")

    print("Q2")
    # Read the predefined probability of a letter occuring in a language
    english_letter_prob_vec, spanish_letter_prob_vec = get_parameter_vectors()

    # Calculating X_1 * log (p_1) -- observed count * log(expected probability)
    total_count = 0
    for letter, count in observed_sorted_freq.items():
        total_count += count
    X_1 = observed_sorted_freq['A']
    # print(english_letter_prob_vec[0])
    print("%.4f" % (X_1 * log(english_letter_prob_vec[0])))
    print("%.4f" % (X_1 * log(spanish_letter_prob_vec[0])))
    
    print("Q3")
    # Using the normalized function with log
    # It is the F(y), which is log(f(y)) = log(P(Y=y)) + sum(x_i * log(p_i))
    # (As we have normalized it by removing C(x) and P(x)), 
    # log(P(Y=y)) is blind, which is the probability of observing english without seeing the evidence
    # p_i is the known probabilities, x_i is the observed count
    # print("\nENGLISH")
    F_English = log(blind_english_prob) + sum_observed_log_prob(observed_sorted_freq, english_letter_prob_vec)

    # print("\nSPANISH")
    F_Spanish = log(blind_spanish_prob) + sum_observed_log_prob(observed_sorted_freq, spanish_letter_prob_vec)
    print("%.4f" % F_English)
    print("%.4f" % F_Spanish)

    print("Q4")
    # Computing P(Y=english | X)
    P_English_Given_Observed_prob = 0
    if F_Spanish - F_English >= 100:
        P_English_Given_Observed_prob = 0
    elif F_Spanish - F_English <= -100:
        P_English_Given_Observed_prob = 1
    else:
        P_English_Given_Observed_prob = 1 / (1 + pow(math.e, F_Spanish - F_English))
    print("%.4f" % P_English_Given_Observed_prob)
