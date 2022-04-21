from TODO_baselines import clean_data
import utils
# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate c_precision, c_recall, and F1 for each class


def evaluate(predictions, true_lables):

    n = " ".join(true_lables).count("N")
    c = " ".join(true_lables).count("C")
    total = n + c

    # This section is as C as positive

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
    #print(f"True pos: {true_pos} \n False pos {false_pos} \n True neg: {true_neg} \n False neg {false_neg}")

    try:
        c_precision = true_pos / (true_pos + false_pos)
    except:
        c_precision = 0
    try:
        c_recall = true_pos / (true_pos / false_neg)
    except:
        c_recall = 0
    try:
        c_f1 = 2 * (c_precision * c_recall) / (c_precision + c_recall)
    except:
        c_f1 = 0

    print(f"C: Precision: {c_precision}. Recall: {c_recall}. F1: {c_f1}.")

    true_pos  = 0
    false_pos = 0
    false_neg = 0
    true_neg  = 0

    # This section is for N as positive
 
    for i in range(0, len(predictions)):  
        labels = true_lables[i].strip().split(" ")   
        pred   = predictions[i].strip().split(" ")  
        for j in range(0, len(pred)):            
            if(labels[j] == pred[j] and pred[j] == "N"):
                true_pos += 1
            if(labels[j] != pred[j] and pred[j] == "N"):
                false_pos += 1
            if(labels[j] == pred[j] and pred[j] == "C"):
                false_neg += 1
            if(labels[j] != pred[j] and pred[j] == "C"):
                true_neg += 1
    #print(f"True pos: {true_pos} \n False pos {false_pos} \n True neg: {true_neg} \n False neg {false_neg}")

    try:
        n_precision = true_pos / (true_pos + false_pos)
    except:
        n_precision = 0
    try:
        n_recall = true_pos / (true_pos / false_neg)
    except:
        n_recall = 0
    try:
        n_f1 = 2 * (n_precision * n_recall) / (n_precision + n_recall)
    except:
        n_f1 = 0

    weighted_f1 = ((c * c_f1) + (n * n_f1)) / total

    print(f"N: Precision: {n_precision}. Recall: {n_recall}. F1: {n_f1}. Weighted F1: {weighted_f1}")
    #First triple C is positive, second N is positive, the last value is the weighted f1
    return (c_precision, c_recall, c_f1), (n_precision, n_recall, n_f1), weighted_f1

if __name__ == '__main__':
    params = utils.Params(f"experiments/base_model/params.json")
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"
    output_path = "output/"
    lstm_path = "experiments/base_model/"

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
    
    with open(lstm_path + f"model_output{params.num_epochs}.txt", encoding="utf8") as LSTM_predictions_file:
        LSTM_predictions = clean_data(LSTM_predictions_file.readlines())

    print("Majority: ")
    majority_eval_c, majority_eval_n, majority_weighted_f1 = evaluate(majority_predictions, dev_labels)

    print("Random: ")
    random_eval_c, random_eval_n, random_weighted_f1 = evaluate(random_predictions, dev_labels)

    print("Length: ")
    length_eval_c, length_eval_n, length_weighted_f1 = evaluate(length_predictions, dev_labels)

    print("Frequency: ")
    frequency_eval_c, frequency_eval_n, frequency_weighted_f1 = evaluate(frequency_predictions, dev_labels)

    print("LSTM: ")
    LSTM_eval_c, LSTM_eval_n, lstm_weighted_f1 = evaluate(LSTM_predictions, test_labels)


    
