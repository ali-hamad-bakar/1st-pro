import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to load MNIST data from ubyte files
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _, num_images, rows, cols = np.fromfile(f, dtype='>u4', count=4)
        images = np.fromfile(f, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        _, num_labels = np.fromfile(f, dtype='>u4', count=2)
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Define the file paths for the MNIST dataset-
train_images_file = 'train-images.idx3-ubyte'
train_labels_file = 'train-labels.idx1-ubyte'
test_images_file = 't10k-images.idx3-ubyte'
test_labels_file = 't10k-labels.idx1-ubyte'

# Load the MNIST dataset
train_images = load_mnist_images(train_images_file)
train_labels = load_mnist_labels(train_labels_file)
test_images = load_mnist_images(test_images_file)
test_labels = load_mnist_labels(test_labels_file)

# 1.Display 5-10 vectorized images with their labels
for i in range(5, 11):
    plt.subplot(2, 3, i-4)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
    plt.axis('off')

    plt.tight_layout()
plt.show()

# 2. Perform vectorization on the images
num_samples = train_images.shape[0]
image_vector_size = train_images.shape[1] * train_images.shape[2]
train_images_vectorized = train_images.reshape(num_samples, image_vector_size)
test_images_vectorized = test_images.reshape(test_images.shape[0], image_vector_size)

# 3. Normalize the vectorized data
train_images_normalized = train_images_vectorized / 255.0
test_images_normalized = test_images_vectorized / 255.0

# 4. Split the data into training and testing sets
ratios = [(0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.01, 0.99)]  # Ratios for splitting
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(
        train_images_normalized, train_labels, test_size=ratio[1], random_state=42
    )

    print(f"Data split ratio: {ratio[0]}:{ratio[1]}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}\n")

    X_train, X_val, y_train, y_val = train_test_split(
    train_images_normalized, train_labels, test_size=0.2, random_state=42
)

# 5. Define and train different classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, classifier in classifiers.items():
    print(f"Training {name}...")
    classifier.fit(X_train, y_train)

# 6. Evaluate the classifier on the validation set
    y_pred = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy}\n")
    
# 7. Evaluate the best performing model on the test set
best_classifier = classifiers["Random Forest"]
best_classifier.fit(train_images_normalized, train_labels)
y_pred_test = best_classifier.predict(test_images_normalized)
test_accuracy = accuracy_score(test_labels, y_pred_test)
print(f"Test Accuracy of the Best Model: {test_accuracy}")

# Train the Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(train_images_normalized, train_labels)

# Make predictions on the test set
y_pred = random_forest.predict(test_images_normalized)

# Evaluate the model's performance
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, average='macro')
recall = recall_score(test_labels, y_pred, average='macro')
f1 = f1_score(test_labels, y_pred, average='macro')
confusion_mat = confusion_matrix(test_labels, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(confusion_mat)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_images_normalized, train_labels, test_size=0.2, random_state=42
)

# Define and train different classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, classifier in classifiers.items():
    print(f"Training {name}...")
    classifier.fit(X_train, y_train)

    # Evaluate the classifier on the validation set
    y_pred = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    results[name] = accuracy
    print(f"Validation Accuracy: {accuracy}\n")

# Compare the results
best_model = max(results, key=results.get)
print(f"Best Model: {best_model} with accuracy {results[best_model]}")

plt.tight_layout()
plt.show()

