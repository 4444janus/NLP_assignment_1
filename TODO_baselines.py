# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from cProfile import label
from hashlib import new
from model.data_loader import DataLoader
import random
from wordfreq import zipf_frequency, word_frequency
import matplotlib.pyplot as plt

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_labels):
      
    """ print("N is the majority class") 
    print("N:", train_labels.count("N"))
    print("C:", train_labels.count("C")) """

    cleaned_labels = clean_data(train_labels)
   
    majority_class = "N"
    predictions = []

    for instance in train_labels:
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        predictions.append(instance_predictions)
    
    print("Majority baseline: ")
    accuracy = Accuracy(predictions, cleaned_labels)

    return accuracy, predictions

def random_baseline(training_labels):    
    SEED = 101

    predictions = []    
    classes = ["N", "C"]
    random.seed(SEED)
    for instance in training_labels:        
        tokens = instance.split(" ")
        predictions_sentence = []
        for label in tokens:
            rand_int = random.randint(0,1)
            predictions_sentence.append(classes[rand_int])
        predictions.append(predictions_sentence)
    
    print("Random baseline: ") # We know that the accuracy should be 0.5
    accuracy = 0.5
    print(f"Accuracy: {accuracy}")
    return accuracy, predictions

def length_baseline(traininput, train_labels):

    training_words_length = []
    label_list = []

    # In this loop we make a list of lists with all the lengths of the words
    # And we reformat the labels
    for i in range(0, len(traininput)):
        labels = train_labels[i].split(" ")
        words  = traininput[i].split(" ")
        length_sentence = []
        label_sentence = []
        for j in range(0, len(words)):  
            length_sentence.append(len(words[j])) 
            label_sentence.append(labels[j])         
            #print((len(words[j]), labels[j]))
        training_words_length.append(length_sentence)
        label_list.append(label_sentence)

    # Try out various length thresholds
    '''for thresh in range(1,20):
        predictions = []
        for i in range(0, len(label_list)):    
            prediction_sentence = []        
            for j in range(0, len(label_list[i])):                  
                if(thresh > training_words_length[i][j]):
                    # Threshold is bigger then the length of the word, so "N"
                    prediction_sentence.append("N")
                else:
                        prediction_sentence.append("C")
            predictions.append(prediction_sentence)
        print(f"Threshold: {thresh}")
        accuracy = Accuracy(predictions, label_list)'''

    # 9 was found to be the best threshold!
    predictions = []
    for i in range(0, len(label_list)):    
        prediction_sentence = []        
        for j in range(0, len(label_list[i])):                  
            if(9 > training_words_length[i][j]):
                # Threshold is bigger then the length of the word, so "N"
                prediction_sentence.append("N")
            else:
                    prediction_sentence.append("C")
        predictions.append(prediction_sentence)

    print("Length baseline:")
    accuracy = Accuracy(predictions, label_list)
    return accuracy, predictions

def frequency_baseline(traininput, train_labels):
    
    frequency = []
    for sentence in traininput:
        frequency_sentence = []
        for word in sentence.split(" "):
            frequency_sentence.append(word_frequency(word, "en"))
        frequency.append(frequency_sentence)

    

    thresholds = []
    accuracies = []

    """ for thresh in range(0, 25, 1):
        new_threshold = thresh * 0.5
        predictions = []
        for i in range(0, len(frequency)):    
            prediction_sentence = []        
            for j in range(0, len(frequency[i])):                  
                if(new_threshold > frequency[i][j]):
                    # Threshold is bigger then the frequency of the word, so "N"
                    prediction_sentence.append("N")
                else:
                        prediction_sentence.append("C")
            predictions.append(prediction_sentence)
        print(f"Threshold: {new_threshold}")
        accuracy = Accuracy(predictions, train_labels) 
        thresholds.append(new_threshold)
        accuracies.append(accuracy) """

        
    new_threshold = 0.000018
    predictions = []
    for i in range(0, len(frequency)):    
        prediction_sentence = []        
        for j in range(0, len(frequency[i])):                  
            if(new_threshold > frequency[i][j]):
                # Threshold is bigger then the frequency of the word, so "C"
                prediction_sentence.append("C")
            else:
                    prediction_sentence.append("N")
        predictions.append(prediction_sentence)
    print("Frequency baseline: ")
    print(f"Threshold: {new_threshold}")
    accuracy = Accuracy(predictions, train_labels) 
    thresholds.append(new_threshold)
    accuracies.append(accuracy)
          

    """ plt.plot(thresholds, accuracies)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.show() """

    return accuracy, predictions

def clean_data(data):
    cleaned_data = []
    for sentence in data:
        cleaned_data.append(sentence.strip())
    return cleaned_data

def save_predictions(predictions, file_name):
    file_string = ""
    for i in range(len(predictions)):
        file_string += " ".join(predictions[i])
        if(i < len(predictions) - 1):
            file_string += "\n"

    with open(file_name, 'w') as f:
        f.write(file_string)
            

def Accuracy(predictions, actual_labels):
    """ true_pos  = 0
    false_pos = 0
    false_neg = 0
    true_neg  = 0 """

    right = 0
    total_length = sum( [ len(listElem) for listElem in predictions])

    for i in range(0, len(predictions)):
        #print("Sent:", i)
        #print("True:", actual_labels, len(actual_labels))
        #print("Pred:", predictions[i], len(predictions))        
        for j in range(0, len(predictions[i])):
            true_label = actual_labels[i][j]
            predicted_label = predictions[i][j]
            if(true_label == predicted_label):
                right += 1
            """ 
            if(true_label == predicted_label and predicted_label == "C"):
                true_pos += 1
            if(true_label != predicted_label and predicted_label == "C"):
                false_pos += 1
            if(true_label == predicted_label and predicted_label == "N"):
                false_neg += 1
            if(true_label != predicted_label and predicted_label == "N"):
                true_neg += 1 """
    print(f"Right: {right} Total: {total_length} Accuracy: {round(right/total_length, 3)}")
    #print(f"True pos: {true_pos} \n False pos {false_pos} \n True neg: {true_neg} \n False neg {false_neg}")
    return right/total_length
    



if __name__ == '__main__':
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"
    output_path = "output/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "sentences.txt", encoding="utf8") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "labels.txt", encoding="utf8") as label_file:
        train_labels = label_file.readlines()


    with open(dev_path + "sentences.txt", encoding="utf8") as dev_file:
        dev_sentences = dev_file.readlines()

    with open(train_path + "labels.txt", encoding="utf8") as dev_label_file:
        dev_labels = dev_label_file.readlines()
    with open(test_path + "sentences.txt") as testfile:
        testinput = testfile.readlines()

    with open(test_path + "labels.txt", encoding="utf8") as test_label_file:
        test_labels = test_label_file.readlines()

    cleaned_train_sentences = clean_data(train_sentences)
    cleaned_train_labels = clean_data(train_labels)
    cleaned_testinput = clean_data(testinput)
    cleaned_test_labels = clean_data(test_labels)

    majority_accuracy, majority_predictions = majority_baseline(cleaned_train_labels)
    random_accuracy, random_predictions     = random_baseline(cleaned_train_labels)
    length_accuracy, length_predictions = length_baseline(cleaned_train_sentences, cleaned_train_labels)
    frequency_accuracy, frequency_predictions = frequency_baseline(cleaned_train_sentences, cleaned_train_labels)

    save_predictions(majority_predictions, output_path + "majority_predictions.txt")
    save_predictions(random_predictions, output_path + "random_predictions.txt")
    save_predictions(length_predictions, output_path + "length_predictions.txt")
    save_predictions(frequency_predictions, output_path + "frequency_predictions.txt")

    # TODO: output the predictions in a suitable way so that you can evaluate them