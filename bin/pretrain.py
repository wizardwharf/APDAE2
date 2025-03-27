from linear_regression import train_model, pickle_model

MODEL_FILE = 'results/model.pkl'

model = train_model()
pickle_model(model, MODEL_FILE)

print('Pretrained model saved to', MODEL_FILE)
