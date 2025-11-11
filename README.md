# Iris Dataset Classification with Decision Tree 🌸

A machine learning project that classifies Iris flower species using a Decision Tree classifier from scikit-learn.

## 📋 Project Overview

This project demonstrates a simple yet effective classification model using the famous Iris dataset. The Decision Tree algorithm is used to predict the species of Iris flowers based on their sepal and petal measurements.

## 🎯 Features

- Loads and preprocesses the Iris dataset
- Creates pandas DataFrame for data exploration
- Trains a Decision Tree classifier
- Evaluates model performance with accuracy metrics
- Visualizes the decision tree structure

## 📊 Dataset

The Iris dataset contains 150 samples of iris flowers with:
- **Features**: 4 measurements (sepal length, sepal width, petal length, petal width)
- **Target**: 3 species (Setosa, Versicolor, Virginica)
- **Samples**: 50 samples per species

## 🛠️ Technologies Used

- **Python 3.x**
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation and analysis
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/<Vinit1235>/Iris_Flower_DecisionTree.git
cd Iris_Flower_DecisionTree
```

2. Install required packages:
```bash
pip install scikit-learn pandas matplotlib seaborn
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the main script:
```bash
python Iris_DecisionTree.py
```

The script will:
1. Load the Iris dataset
2. Display the first few rows of the dataset
3. Split data into training (80%) and testing (20%) sets
4. Train the Decision Tree model
5. Print the accuracy score
6. Display a visualization of the decision tree

## 📈 Results

The Decision Tree classifier typically achieves **95-100% accuracy** on the test set, demonstrating excellent performance on this well-structured dataset.

## 🔍 Code Structure

- Data loading and preprocessing
- One-hot encoding of species labels
- Train-test split (80-20)
- Model training with `DecisionTreeClassifier`
- Performance evaluation
- Tree visualization

## 📝 Example Output

```
Accuracy: 1.0
```

The model also generates a visual representation of the decision tree showing the decision rules at each node.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

[Your Name] - [GitHub Profile](https://github.com/<your-username>)

## 🙏 Acknowledgments

- Iris dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- scikit-learn documentation and community

---

⭐ Star this repository if you find it helpful!
