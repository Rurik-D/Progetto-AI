from model_trainer.digits_model import generate_model, test_labels_data

train_data, test_data, train_labels, test_labels = test_labels_data()

generate_model(train_data, test_data, train_labels, test_labels)
