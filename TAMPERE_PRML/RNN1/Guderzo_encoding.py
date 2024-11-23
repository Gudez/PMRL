import numpy as np

def convert_to_hot(filename=""):
    """
    Given a file path, it convert the txt file into a one hot encoded matrix
    :param filename: file txt path
    :return: one_hot_encoded_matrix
    """
    # Load ascii text and covert everything to lowercase
    # I take into account the space " ", exlude the tab \n
    with open(filename, "r", encoding='utf-8') as f:
        raw_text = f.read()

    # Unique letters in the file
    chars = sorted(list(set(raw_text)))
    num_chars = len(chars)

    print(f"Total number of characters: {len(raw_text)}")
    print(f"Size of vocabulary: {num_chars}")

    # First convert characters to integers by Python dictionary
    # One-hot and dictionaries from char to int and viceversa
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # One-hot encode each word in the raw text
    one_hot_encoded_list = []
    for word in raw_text:
        # Convert the word to a list of character indices using the dictionary
        char_indices = [char_to_int[char] for char in word]

        # Initialize a one-hot encoded matrix for the word
        one_hot_encoded_word = np.zeros((len(word), len(chars)))

        # Set the corresponding indices to 1
        one_hot_encoded_word[np.arange(len(word)), char_indices] = 1

        # Append the one-hot encoded word to the list
        one_hot_encoded_list.append(one_hot_encoded_word)

    # Stack the list of one-hot words along the first axis to create the matrix
    one_hot_encoded_matrix = np.stack(one_hot_encoded_list).reshape(len(raw_text), num_chars)

    # Sanity check: from one hot encoded to integer, from integer to char
    # foo = np.argmax(one_hot_encoded_matrix,axis=1)
    # print('Sanity check')
    # print(one_hot_encoded_matrix[0:19])
    # print(foo[0:19]) # First row of the file as int
    # print(''.join([int_to_char[f] for f in foo[0:19]]), end="") # First row of the file

    return one_hot_encoded_matrix, char_to_int, int_to_char
def main():
    filepath = "C:/Users/ricca/Desktop/RGud/TAMPERE_UNIVERSITY/2_SEMESTER/PATTERN RECOGNITION AND MACHINE LEARNING/EXERCISE5/abcde.txt"
    convert_to_hot(filename=filepath)

if __name__ == "__main__":
    main()