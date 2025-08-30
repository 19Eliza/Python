***1. Data exploration and correlation analysis***

The classic Iris dataset was imported from the UCI repository and transformed into a structured table.

Performed correlation analysis between all numerical features (sepal length, sepal width, petal length, petal width) to identify relationships between them.

Additionally, correlations were calculated within each class (Setosa, Versicolor, Virginica) to see how patterns differ depending on species.

Constructed visualizations: histograms and pairwise scatterplots showing how each class is distributed across feature space, allowing us to visually assess separability.

***2. Comparative study of classification methods***

Chose a subset of the dataset consisting of only two features for clarity of visualization.

Trained multiple classification models:

Linear Discriminant Analysis (LDA),

Quadratic Discriminant Analysis (QDA),

Logistic Regression,

Support Vector Machines (SVM) with linear kernel,

SVM with polynomial (quadratic) kernel.

For each model, decision boundaries (separating curves) were plotted on the feature space, demonstrating how different algorithms partition the data and highlighting their strengths and weaknesses.

***3. Focused analysis on the hardest-to-separate features***

Identified the two least separable variables (those with the strongest overlap between Versicolor and Virginica).

Built a linear discriminant classifier specifically on these features to test how well it distinguishes the two classes in this challenging scenario.

Visualized both the classifier’s decision boundary and the actual class distributions, showing regions of correct classification and zones where misclassification occurs.

This experiment helps to understand the limitations of linear models when classes overlap heavily in feature space.