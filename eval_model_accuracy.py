import Generate_Data
from question_classifier import QuestionClassifier

def eval_model_accuracy():
    question_generator = Generate_Data.Gen_Data()
    generated_data, _ = question_generator.generate()

    classifier = QuestionClassifier('question_set_clean.csv', use_new=True)

    total = 0
    samples = 0
    for data in generated_data:
        expect_question = data[0]
        for i in range(1, len(data)):
            samples += 1
            test = data[i]
            classified_question = classifier.classify_question(test)
            if classified_question != expect_question:
                print("--------")
                print("Question: {}".format(test))
                print("Expected to be classified as: {}".format(expect_question))
                print("Got classification: {}".format(classified_question))
                print("-------")
            else:
                total += 1

    print("Accuracy: {}".format(float(total) / samples))
    print(generated_data)

if __name__ == "__main__":
    eval_model_accuracy()

