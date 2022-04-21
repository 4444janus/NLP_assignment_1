from TODO_baselines import clean_data
# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class

def evaluate(predictions, true_lables):
    true_pos  = 0
    false_pos = 0
    false_neg = 0
    true_neg  = 0
 
    for i in range(0, len(predictions)):  
        labels = true_lables[i].strip().split(" ")   
        pred   = predictions[i].strip().split(" ")  
        for j in range(0, len(pred)):            
            if(labels[j] == pred[j] and pred[j] == "C"):
                true_pos += 1
            if(labels[j] != pred[j] and pred[j] == "C"):
                false_pos += 1
            if(labels[j] == pred[j] and pred[j] == "N"):
                false_neg += 1
            if(labels[j] != pred[j] and pred[j] == "N"):
                true_neg += 1
    print(f"True pos: {true_pos} \n False pos {false_pos} \n True neg: {true_neg} \n False neg {false_neg}")

    try:
        precision = true_pos / (true_pos + false_pos)
    except:
        precision = 0
    try:
        recall = true_pos / (true_pos / false_neg)
    except:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = 0

    print(f"Precision: {precision}. Recall: {recall}. F1: {f1}.")



if __name__ == '__main__':
    
    dev_path = "NLP_assignment_1/data/preprocessed/val/"
    test_path = "NLP_assignment_1/data/preprocessed/test/"
    output_path = "NLP_assignment_1/output/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.
    
    with open(dev_path + "labels.txt", encoding="utf8") as dev_label_file:
        dev_labels = dev_label_file.readlines()

    with open(test_path + "labels.txt", encoding="utf8") as test_label_file:
        test_labels = clean_data(test_label_file.readlines())

    with open(output_path + "majority_predictions.txt", encoding="utf8") as majority_predictions_file:
        majority_predictions = clean_data(majority_predictions_file.readlines())

    with open(output_path + "random_predictions.txt", encoding="utf8") as random_predictions_file:
        random_predictions = clean_data(random_predictions_file.readlines())

    with open(output_path + "length_predictions.txt", encoding="utf8") as length_predictions_file:
        length_predictions = clean_data(length_predictions_file.readlines())

    with open(output_path + "frequency_predictions.txt", encoding="utf8") as frequency_predictions_file:
        frequency_predictions = clean_data(frequency_predictions_file.readlines())

print("Majority: ")
evaluate(majority_predictions, dev_labels)

print("Random: ")
evaluate(random_predictions, dev_labels)

print("Length: ")
evaluate(length_predictions, dev_labels)

print("Frequency: ")
evaluate(frequency_predictions, dev_labels)

#print("LSTM: ")
#evaluate(majority_predictions, dev_labels)

    
