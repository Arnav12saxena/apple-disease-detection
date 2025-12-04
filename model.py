import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image

plt.style.use('ggplot')  
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 6),  
    'axes.grid': True,
    'grid.alpha': 0.3,
})

classes = ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']
num_classes = len(classes)

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

def create_labels(label_str):
    labels = [0] * num_classes
    for lab in label_str.split():
        if lab in classes:
            labels[classes.index(lab)] = 1
    return np.array(labels, dtype=np.float32)

train_df = pd.read_csv('train.csv')
train_image_paths = ['train_images/' + img for img in train_df['image']]
train_labels = [create_labels(label) for label in train_df['labels']]

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_image_paths, train_labels, test_size=0.2, random_state=42
)

test_image_paths = ['test_images/' + img for img in os.listdir('test_images')]
test_image_names = os.listdir('test_images')

def create_dataset(image_paths, labels=None, is_train=True):
    if is_train or labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(lambda x, y: (load_image(x), y),
                             num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(lambda x: load_image(x),
                             num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_paths, train_labels, is_train=True)
val_dataset = create_dataset(val_paths, val_labels, is_train=False)
test_dataset = create_dataset(test_image_paths, is_train=False)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

num_epochs = 10  
history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    verbose=1
)

final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
with open('accuracy_metrics.txt', 'w') as f:
    f.write(f'Final Training Accuracy: {final_train_accuracy:.4f}\n')
    f.write(f'Final Validation Accuracy: {final_val_accuracy:.4f}\n')
print(f'Final training accuracy: {final_train_accuracy:.4f}')
print(f'Final validation accuracy: {final_val_accuracy:.4f}')
print('Accuracy metrics saved to accuracy_metrics.txt')

model.save('apple_disease_model.h5')
print('Model saved as apple_disease_model.h5')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='navy', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='darkorange', linewidth=2)
plt.title('Training and Validation Loss', fontsize=18, pad=15)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_curves.png', bbox_inches='tight', dpi=300)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='navy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='darkorange', linewidth=2)
plt.title('Training and Validation Accuracy', fontsize=18, pad=15)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('accuracy_curves.png', bbox_inches='tight', dpi=300)

plt.close()
print('Training and validation curves saved as loss_curves.png and accuracy_curves.png')

val_images = np.concatenate([batch[0] for batch in val_dataset], axis=0)
val_true = np.concatenate([batch[1] for batch in val_dataset], axis=0)
val_pred_probs = model.predict(val_images)
val_pred = (val_pred_probs > 0.5).astype(np.int32)

for i, cls in enumerate(classes):
    cm = confusion_matrix(val_true[:, i], val_pred[:, i])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='viridis', ax=ax)
    ax.set_title(f'Confusion Matrix for {cls}', fontsize=16, pad=15)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    for text in ax.texts:
        text.set_fontsize(12)
    plt.savefig(f'confusion_matrix_{cls}.png', bbox_inches='tight', dpi=300)
    plt.close()

print('Confusion matrices saved as confusion_matrix_<class>.png for each class')

num_visualizations = 10
val_paths_sample = val_paths[:num_visualizations]
val_labels_sample = val_labels[:num_visualizations]
val_images_sample = np.array([np.array(Image.open(path).convert('RGB')) for path in val_paths_sample])  

val_pred_sample = val_pred[:num_visualizations]

for idx in range(num_visualizations):
    img = val_images_sample[idx]
    true_lbls = [classes[j] for j in range(num_classes) if val_labels_sample[idx][j] == 1]
    pred_lbls = [classes[j] for j in range(num_classes) if val_pred_sample[idx][j] == 1]
    
    correct = set(true_lbls) & set(pred_lbls)
    wrong = set(pred_lbls) - set(true_lbls)
    missed = set(true_lbls) - set(pred_lbls) 
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    
    y_pos = 30
    x_pos = 10
    font_size = 14
    line_height = 35
    
    if correct:
        plt.text(x_pos, y_pos, 'Correct: ' + ', '.join(correct), color='green', fontsize=font_size, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='green', boxstyle='round,pad=0.5'))
        y_pos += line_height
    if wrong:
        plt.text(x_pos, y_pos, 'Wrong: ' + ', '.join(wrong), color='red', fontsize=font_size, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.5'))
        y_pos += line_height
    if missed:
        plt.text(x_pos, y_pos, 'Missed: ' + ', '.join(missed), color='red', fontsize=font_size, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.5'))
    
    plt.title(f'Prediction for Image {idx+1}', fontsize=16, pad=15)
    plt.savefig(f'prediction_image_{idx+1}.png', bbox_inches='tight', dpi=300)
    plt.close()

print(f'{num_visualizations} prediction images saved as prediction_image_<num>.png with green for correct, red for wrong/missed')

predictions = []
for img_path, img_name in zip(test_image_paths, test_image_names):
    img = load_image(img_path)
    img = tf.expand_dims(img, axis=0)  
    probs = model.predict(img, verbose=0)[0]
    pred_labels = [classes[j] for j in range(num_classes) if probs[j] > 0.5]
    predictions.append({'image': img_name, 'labels': ' '.join(pred_labels) or 'healthy'}) 

sub_df = pd.DataFrame(predictions)
sub_df.to_csv('submission.csv', index=False)
print('Submission file created as submission.csv')